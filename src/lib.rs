// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![allow(clippy::bool_assert_comparison)]

//! Doubly-linked intrusive lists for scheduling and waking.
//!
//! A [`WaitList<T>`][WaitList] keeps track of nodes that each contain some
//! value `T`. The list is kept in ascending sorted order by comparing the `T`s:
//!
//! - [`WaitList::wait`] produces a future that, when polled, will traverse the
//!   list and insert a node in the proper place, then wait for the node to be
//!   kicked back out.
//!
//! - [`WaitList::wake_le`] starts at one end and removes every node with a
//!   value less than a threshold.
//!
//! The sort order is intended for keeping track of timestamps/deadlines, but
//! you may find others uses for it.
//!
//! If you just want to keep things in a list, and don't care about order or
//! need to associate a timestamp, simply use `WaitList<()>`. This disables the
//! sorting and removes the order-related fields from both the list and node.
//!
//! # How to use for sleep/wake
//!
//! The basics are straightforward: given a `WaitList` tracking waiters on a
//! particular event, create a wait future by calling `wait` on it. At some
//! future point in a concurrent process or interrupt handler, one of the
//! `wake_*` methods on `WaitList` gets called, and the node will be removed and
//! its associated `Waker` invoked, causing the future produced by `wait` to
//! resolve.
//!
//! If you're using an async runtime, you probably don't need to think about
//! `Waker`s specifically -- they're typically managed by the runtime.
//!
//! # Pinning
//!
//! Because `WaitList` creates circular, self-referential data structures, all
//! operations require that they be
//! [pinned](https://doc.rust-lang.org/core/pin/). Because we don't use the
//! heap, we provide ways to create and use pinned data structures on the stack.
//!
//! Here is an example of creating a list and then waiting on it using async
//! code:
//!
//! ```rust
//! # async fn foo() {
//! lilist::create_list!(wait_list);
//!
//! wait_list.wait(()).await;
//! # }
//! ```
//!
//! # How is this safe/sound?
//!
//! Doubly-linked lists aren't a very popular data structure in Rust because
//! it's very, very hard to get them right. I believe this implementation to be
//! sound because it relies on *blocking*.
//!
//! - The list node is opaquely contained in the future returned by `wait`.
//!
//! - Calling `wait` only returns a future that captures a pinned reference to
//! the list (which borrows it, preventing it from going anywhere). 
//!
//! - To actually insert itself into the list, the future requires that it be
//! polled, which requires in turn that it be pinned, assuring us that _it_ is
//! also not going anywhere.
//!
//! - Due to the contract required by `Pin`, we also can be confident that the
//! future (and thus the node) won't be overwritten or deallocated without being
//! dropped, and its `Drop` impl will unwire it from the list.
//!
//! This is why there is no `insert` operation, or a `take` operation that
//! returns a node -- both operations would compromise memory safety.

// Implementation safety notes:
//
// The safety comments in this module reference the following invariants:
//
// Link Valid Invariant: all the link pointers (that is, `Node::prev` and
// `Node::next`) transitively reachable from either `WaitList` or `Node` are
// valid / not dangling. We maintain this by only setting them to the addresses
// of pinned structures, and ensuring that the `Drop` impl of those pinned
// structures will remove their addresses from any link.

#![cfg_attr(not(test), no_std)]

#![warn(
    elided_lifetimes_in_paths,
    explicit_outlives_requirements,
    missing_debug_implementations,
    missing_docs,
    semicolon_in_expressions_from_macros,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_op_in_unsafe_fn,
    unreachable_pub,
    unused_qualifications,
)]

use core::cell::{Cell, UnsafeCell};

use core::fmt::Debug;
use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;
use core::task::{Poll, RawWaker, RawWakerVTable, Waker};
use core::marker::{PhantomData, PhantomPinned};

use pin_project::{pin_project, pinned_drop};

/// Marker trait implementing the "Captures Trick" from Rust RFC 3498, ensuring
/// that we do lifetime capturing right in the 2021 edition.
///
/// TODO: revisit this when we can switch to the 2024 edition, where the default
/// behavior makes this less necessary.
pub trait Captures<T> {}

impl<U: ?Sized, T> Captures<T> for U {}


///////////////////////////////////////////////////////////////////////////
// Node implementation

/// A node that can be inserted into a [`WaitList`] and used to wait for an
/// event.
struct Node<T> {
    /// Links to the previous and next things in a list, in that order.
    ///
    /// If this node is not a member of a list (is "detached"), this will be
    /// `None`.
    links: Cell<Option<(LinkPtr<T>, LinkPtr<T>)>>,
    /// Waker to poke when this node is kicked out of a list.
    waker: Cell<Option<Waker>>,
    /// Value used to order this node in a list.
    contents: T,

    _marker: (NotSendMarker, PhantomPinned),
}

impl<T> Node<T> {
    /// Creates a new `Node` containing `contents` as its value.
    fn new(contents: T) -> Self {
        Self {
            links: Cell::default(),
            waker: Cell::new(None),
            contents,
            _marker: (NotSendMarker::default(), PhantomPinned),
        }
    }

    /// Disconnects a node from any list. This is idempotent.
    ///
    /// This does _not_ poke the waker, but returns it if one was present (which
    /// should occur exactly when this node is in a list).
    fn detach(self: Pin<&Self>) -> Option<Waker> {
        // TODO: it's not clear that this operation needs to require Pin. It's
        // safe if called on a non-pinned node, since by definition if a node is
        // not pinned it isn't in a list.

        if let Some((prev, next)) = self.links.take() {
            // Safety: per the Link Valid Invariant we can dereference these
            // pointers, which is the property change_{next,prev} need to be
            // used safely.
            unsafe {
                prev.change_next(next);
                next.change_prev(prev);
            }
            self.waker.take()
        } else {
            None
        }
    }

    /// Checks if a node is detached.
    fn is_detached(&self) -> bool {
        self.links.get().is_none()
    }
}

/// A `Node` will check that it isn't still in a list on drop, and panic if it
/// is. This shouldn't be possible if the wait future is correct, since dropping
/// a node means that it's not currently owned by a list insertion future.
impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        assert!(self.is_detached());
    }
}

/// We have to write a custom `Debug` impl for this type because `Waker` doesn't
/// impl `Debug`.
impl<T: Debug> Debug for Node<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Node")
            .field("links", &self.links)
            .field("waker", &"...")
            .field("contents", &self.contents)
            .finish()
    }
}

///////////////////////////////////////////////////////////////////////////
// WaitList implementation

/// Shorthand for `PinPtr<Node<T>>`
type Npp<T> = PinPtr<Node<T>>;

/// A list of waiters, waiting for something.
///
/// The list *references*, but does not *own*, the nodes. The nodes are owned by
/// the wait futures (returned by [`wait`]/[`wait_with_cleanup`]), and the
/// creator of those futures own them. This is okay because the futures must be
/// pinned to join the list, and this means we can be sure their drop impls
/// (which will detach the nodes from the list) will run before the memory is
/// repurposed or moved.
///
/// Because lists contain self-referential pointers, creating one is somewhat
/// involved. Use the [`create_list!`] macro when possible, or see
/// `WaitList::new` for instructions.
///
/// # Drop
///
/// You must remove/wake all the nodes in a list before dropping the list.
/// Dropping a list without emptying it is treated as a programming error, and
/// will panic.
///
/// That being said, because the wait-futures that keep nodes in the list borrow
/// the list, it should be impossible to drop the list while there's anything in
/// it.
pub struct WaitList<T> {
    links: Cell<Option<(Npp<T>, Npp<T>)>>,
    _marker: (NotSendMarker, PhantomPinned),
}

#[allow(clippy::new_without_default)]
impl<T> WaitList<T> {
    /// Creates a `WaitList` in an initialized, empty state.
    ///
    /// In this state, the list is not very useful. To access most of its API,
    /// you must pin it. The simplest way to do this properly is with the
    /// `pin!` macro, though because it assumes you want an exclusive reference
    /// to the pinned value, you'll also want to call `Pin::into_ref`, like so:
    ///
    /// ```
    /// # use lilist::WaitList;
    /// // create the list on the stack
    /// let list = core::pin::pin!(WaitList::<()>::new());
    /// // drop exclusivity
    /// let list = list.into_ref();
    /// ```
    ///
    /// This crate provides a `create_list!` macro that handles this for you, if
    /// you'd prefer:
    ///
    /// ```
    /// # use lilist::create_list;
    /// create_list!(list, ());
    /// ```
    pub fn new() -> WaitList<T> {
        Self {
            links: Cell::default(),
            _marker: (NotSendMarker::default(), PhantomPinned),
        }
    }
}

/// Operations on lists of ordered nodes (which includes `()`).
impl<T: PartialOrd> WaitList<T> {
    /// Produces a future that, when polled, will insert a node into this list
    /// just after any other nodes that are `<= contents`.
    ///
    /// (For the simplest case where `T` is `()`, this means the node will be
    /// inserted at the end of the list.)
    ///
    /// # Cancellation
    ///
    /// Dropping this future immediately does effectively nothing.
    ///
    /// Dropping this future after poll, but before it resolves, removes the
    /// node from the list.
    pub fn wait<'list>(
        self: Pin<&'list Self>,
        contents: T,
    ) -> impl Future<Output = ()> + Captures<&'list Self> {
        self.wait_with_cleanup(contents, || ())
    }

    /// Produces a future that, when polled, will insert a node into this list
    /// just after any other nodes that are `<= contents`.
    ///
    /// (For the simplest case where `T` is `()`, this means the node will be
    /// inserted at the end of the list.)
    ///
    /// This version has a "cleanup action" that will be run if the node is
    /// detached from the list, and then dropped before this future is polled to
    /// observe it. This can be useful for cleaning up certain kinds of
    /// synchronization structures.
    ///
    /// # Cancellation
    ///
    /// Dropping this future immediately does effectively nothing.
    ///
    /// Dropping this future after poll, but before it resolves, removes the
    /// node from the list. If the node was _already_ removed from the list by
    /// some other code waking it, the `cleanup` function will be called.
    pub fn wait_with_cleanup<'list, F: FnOnce()>(
        self: Pin<&'list Self>,
        contents: T,
        cleanup: F,
    ) -> impl Future<Output = ()> + Captures<&'list Self> {
        WaitForDetach {
            node: UnsafeCell::new(Node::new(contents)),
            list: self,
            state: Cell::new(WaitState::NotYetAttached),
            cleanup: Cell::new(Some(cleanup)),
        }
    }

    /// Beginning at the head of the list, removes each node `n` where
    /// `n.contents <= threshold`. Each removed node's waker will be poked.
    ///
    /// After this completes:
    ///
    /// - Any `Node` previously inserted into this list with `contents <=
    ///   threshold` is now detached, and its waker has been called.
    ///
    /// - All `Node`s remaining in this list have `contents > threshold`.
    pub fn wake_le(self: Pin<&Self>, threshold: T) {
        self.wake_while(|t| t <= &threshold)
    }

    /// Work through the list from the head, waking the longest prefix of nodes
    /// that all pass `pred`. That is, applies `pred` to the contents of each
    /// node, waking the node and continuing if it returns `true`, and leaving
    /// the node and stopping if it returns `false`.
    ///
    /// Returns `true` if at least one node was woken up.
    pub fn wake_while(self: Pin<&Self>, mut pred: impl FnMut(&T) -> bool) {
        // Work through nodes from the head (least) moving up.
        let mut candidate = self.links.get().map(|(_t, h)| h);
        while let Some(cptr) = candidate {
            // Safety: Link Valid Invariant
            let cref = unsafe { cptr.get() };
            if !pred(&cref.contents) {
                break;
            }
            // Copy the next pointer before detaching, since it's about to go
            // away.
            let next = cref.links.get().expect("node in list without links").1;
            if let Some(waker) = cref.detach() {
                waker.wake();
            }

            candidate = next.as_node();
        }
    }

    /// Checks whether this list has no waiters on it.
    pub fn is_empty(&self) -> bool {
        self.links.get().is_none()
    }
}

impl<T> WaitList<T> {
    /// Wakes the head of the list, but only if it exists and satisfies `pred`.
    ///
    /// Returns `true` if a node was woken up, `false` otherwise.
    pub fn wake_one_if(self: Pin<&Self>, pred: impl FnOnce(&T) -> bool) -> bool {
        if let Some((_back, candidate)) = self.links.get() {
            // Safety: Link Valid Invariant
            let cref = unsafe { candidate.get() };
            if pred(&cref.contents) {
                if let Some(waker) = cref.detach() {
                    waker.wake();
                }
                return true;
            }
        }
        false
    }

    /// Convenience method for waking all the waiters, because not all ordered
    /// types have an easily available MAX element, and because (on
    /// insertion-ordered queues) `wake_le(())` looks weird.
    pub fn wake_all(self: Pin<&Self>) {
        let mut candidate = self.links.get().map(|(_t, h)| h);
        while let Some(cptr) = candidate {
            // Safety: Link Valid Invariant
            let cref = unsafe { cptr.get() };
            // Copy the next pointer before detaching, since it's about to go
            // away.
            let next = cref.links.get().expect("node in list without links").1;
            if let Some(waker) = cref.detach() {
                waker.wake();
            }

            candidate = next.as_node();
        }
    }
}

/// Operations specific to insertion-orded queues.
impl WaitList<()> {
    /// Wakes the oldest (earliest inserted) waiter on an unsorted list.
    ///
    /// Returns a flag indicating whether anything was done (i.e. whether the
    /// list was non-empty).
    pub fn wake_oldest(self: Pin<&Self>) -> bool {
        self.wake_one_if(|_| true)
    }
}

impl<T> Drop for WaitList<T> {
    fn drop(&mut self) {
        if self.links.get().is_some() {
            panic!();
        }
    }
}

/// We need a custom `Debug` impl because the inferred one will require `T:
/// Debug`; since we only print pointers to `T` we don't need to be so
/// stringent.
impl<T: Debug> Debug for WaitList<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WaitList")
            .field("links", &self.links)
            .finish()
    }
}

/// Creates a pinned list on the stack.
///
/// `create_list!(ident)` is equivalent to `let ident = ...;` -- it creates a
/// local variable called `ident`, holding an initialized list.
#[macro_export]
macro_rules! create_list {
    ($var:ident, $t:ty) => {
        let $var = core::pin::pin!($crate::WaitList::<$t>::new());
        // Drop mutability, since we expect the list to be aliased shortly.
        let $var = $var.into_ref();
    };
    ($var:ident) => {
        $crate::create_list!($var, _)
    };
}

///////////////////////////////////////////////////////////////////////////
// WaitForDetach future implementation.

/// Internal future type used for `wait`.
#[pin_project(PinnedDrop)]
struct WaitForDetach<'list, T, F: FnOnce()> {
    /// The Node we use to join the list, once we decide to do that. There are a
    /// couple odd things going on here that are worth pointing out.
    ///
    /// First: the Node is in an `UnsafeCell`. This is an attempt at "reference
    /// breaking" to ensure that holding a `Pin<&mut WaitForDetach2>` does not
    /// imply the possibility of generating a `Pin<&mut Node>`. This is
    /// important, because while the Node is in a list, it's referenced by the
    /// link pointers, and code for e.g. waking a node in the list will generate
    /// ephemeral `&` references to the Node. This would imply aliasing the
    /// implied `&mut` path to the Node from `Pin<&mut WaitForDetach2>`, which
    /// would be bad. So, `UnsafeCell`.
    ///
    /// Second: despite being in an `UnsafeCell`, this is still tagged as
    /// `#[pin]`. This is a second safety measure, since it prevents us from
    /// getting a direct `&mut UnsafeCell` out of the `project` operation, which
    /// could let us overwrite it. Instead we get a `Pin<&mut UnsafeCell>` which
    /// we immediately degrade to a `&UnsafeCell` before re-pinning its contents
    /// as a `Pin<&Node<T>>`. This is not a soundness issue so much as a "make
    /// internal implementation harder to break during maintenance" feature.
    #[pin]
    node: UnsafeCell<Node<T>>,
    /// Reference to the list we're borrowing. This reference serves two
    /// purposes.
    ///
    /// First, we use it on first poll to insert our `Node` into the list.
    ///
    /// Second, it ensures that we borrow the list for our entire existence,
    /// which makes it impossible to drop a list while nodes are in it.
    list: Pin<&'list WaitList<T>>,
    /// Tracking poll state.
    state: Cell<WaitState>,
    /// Cleanup action; this is always set to `Some`, but we `take` it to run it
    /// for simplicity, so this becomes `None` during drop, sometimes.
    cleanup: Cell<Option<F>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum WaitState {
    NotYetAttached,
    Attached,
    DetachedAndPolled,
}

impl<T, F: FnOnce()> Future for WaitForDetach<'_, T, F>
    where T: PartialOrd,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut core::task::Context<'_>)
        -> Poll<Self::Output>
    {
        let p = self.as_ref().project_ref();
        // Ensure that we can only produce a Pin<&> to the node.
        //
        // Safety: this is unsafe for two reasons. First, we are dereferencing a
        // raw pointer to access the contents of the UnsafeCell, which
        // effectively bypasses the borrow checker. We only ever produce
        // _shared_ references into the UnsafeCell, making this okay.
        // Second, we are pinning the result. We are basically implementing our
        // own pin projection through the UnsafeCell, converting a `Pin<&mut
        // UnsafeCell<T>>` to a `Pin<&T>`. 
        let node = unsafe { Pin::new_unchecked(&*p.node.get()) };
        match p.state.get() {
            WaitState::NotYetAttached => {
                // Do the insertion part. This used to be a separate `insert` function,
                // but that function had soundness risks and so I've inlined it.

                let nnn = PinPtr::new(node);
                {
                    // Node should not already belong to a list.
                    assert!(node.is_detached());

                    // Work through the nodes starting at the tail, looking for the
                    // future `prev` of `node`.
                    let mut candidate = p.list.links.get().map(|(tail, _)| tail);
                    while let Some(cptr) = candidate {
                        // Safety: Link Valid Invariant means we can deref this
                        let cref = unsafe { cptr.get() };

                        if cref.contents <= node.contents {
                            break;
                        }
                        candidate = match cref.links.get() {
                            Some((LinkPtr::Inner(prev), _)) => Some(prev),
                            _ => None,
                        };
                    }

                    if let Some(neighbor) = candidate {
                        // We must insert just after neighbor.
                        // Safety: Link Valid Invariant means we can get a shared
                        // reference to neighbor's pointee.
                        let nref = unsafe { neighbor.get() };
                        debug_assert!(nref.contents <= node.contents);
                        let (neigh_prev, neigh_next) = nref.links.get().unwrap();
                        node.links.set(Some((LinkPtr::Inner(neighbor), neigh_next)));
                        nref.links.set(Some((neigh_prev, LinkPtr::Inner(nnn))));
                        // Safety: Link Valid Invariant means we can get the shared
                        // reference to neigh_next's pointee that we need to store this
                        // pointer.
                        unsafe {
                            neigh_next.change_prev(LinkPtr::Inner(nnn));
                        }
                    } else {
                        // The node is becoming the new head.
                        if let Some((tail, old_head)) = p.list.links.get() {
                            node.links.set(Some((
                                LinkPtr::End(PinPtr::new(p.list.as_ref())),
                                LinkPtr::Inner(old_head),
                            )));
                            p.list.links.set(Some((tail, nnn)));
                            // Safety: Link Valid Invariant means we can get the shared
                            // reference to old_tail's pointee that we need to store
                            // this pointer.
                            unsafe {
                                LinkPtr::Inner(old_head).change_prev(LinkPtr::Inner(nnn));
                            }
                        } else {
                            // The node is, in fact, becoming the entire list.
                            let lp = LinkPtr::End(PinPtr::new(p.list.as_ref()));
                            node.links.set(Some((lp, lp)));
                            p.list.links.set(Some((nnn, nnn)));
                        }
                    }
                }
                p.state.set(WaitState::Attached);
                node.waker.set(Some(cx.waker().clone()));
                Poll::Pending
            }
            WaitState::Attached => {
                // See if we've detached.
                if node.is_detached() {
                    // The node is not attached to any list, but we're still borrowing
                    // it until we're dropped, so we don't need to replace the node
                    // field contents -- just set a flag to skip work in the Drop impl.
                    p.state.set(WaitState::DetachedAndPolled);
                    Poll::Ready(())
                } else {
                    // The node remains attached to the list. While unlikely, it's
                    // possible that the waker has changed. Update it.
                    node.waker.set(Some(cx.waker().clone()));
                    Poll::Pending
                }
            }
            // This effectively "fuses" the future.
            WaitState::DetachedAndPolled => Poll::Ready(()),
        }
    }
}

#[pinned_drop]
impl<T, F: FnOnce()> PinnedDrop for WaitForDetach<'_, T, F> {
    fn drop(self: Pin<&mut Self>) {
        let p = self.as_ref().project_ref();
        // Ensure that we can only produce a Pin<&> to the node.
        //
        // Safety: this is unsafe for two reasons. First, we are dereferencing a
        // raw pointer to access the contents of the UnsafeCell, which
        // effectively bypasses the borrow checker. We only ever produce
        // _shared_ references into the UnsafeCell, making this okay.
        // Second, we are pinning the result. We are basically implementing our
        // own pin projection through the UnsafeCell, converting a `Pin<&mut
        // UnsafeCell<T>>` to a `Pin<&T>`. 
        let node = unsafe { Pin::new_unchecked(&*p.node.get()) };
        if p.state.get() == WaitState::Attached {
            if node.is_detached() {
                // Uh oh, we have not had a chance to handle the detach.
                if let Some(cleanup) = p.cleanup.take() {
                    cleanup();
                }
            } else {
                // If _we_ detach ourselves, we don't run the cleanup
                // action.
                node.detach();
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Shared utility bits.

/// Zero-sized marker type that can be included to ensure that a data structure
/// is not automatically made `Send` (i.e. safe for transfer across threads).
///
/// This also blocks `Sync`.
#[derive(Default, Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct NotSendMarker(PhantomData<*const ()>);

/// A non-null pointer that points to something pinned.
///
/// This can only be created from a pinned reference, proving that, at the time
/// it was created, the referend was pinned. To dereference it safely, the
/// caller must ensure the referend is _still_ pinned, through whatever means.
struct PinPtr<T>(NonNull<T>);

impl<T> Copy for PinPtr<T> {}

impl<T> Clone for PinPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Debug for PinPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("PinPtr").field(&self.0.as_ptr()).finish()
    }
}

impl<T> PartialEq for PinPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for PinPtr<T> {}

impl<T> PinPtr<T> {
    fn new(r: Pin<&T>) -> Self {
        Self(NonNull::from(&*r))
    }

    /// Dereferences the pointer and pins the result.
    ///
    /// # Safety
    ///
    /// This is safe as long as you know that the referend is still alive and
    /// pinned, through whatever context-specific means you have available.
    unsafe fn get(&self) -> Pin<&T> {
        // Safety: this dereferences our pointer, which is safe as long as the
        // caller knows it's valid by some other means.
        let r = unsafe {
            self.0.as_ref()
        };
        // Safety: this claims the result is pinned, which is safe as long as
        // the caller knows the referend is still pinned, through some other
        // means.
        unsafe {
            Pin::new_unchecked(r)
        }
    }
}

/// A link pointer stored in a node.
///
/// We implement a circular doubly-linked list, but unlike most such lists, we
/// don't play type tricks with the pointers at the end. This enum explicitly
/// records whether each pointer is pointing at a node, or the list root.
///
/// This is more expensive than the type-tricks version (by one word per link),
/// but gives greater assurance against memory safety bugs.
enum LinkPtr<T> {
    /// A pointer to a node within the list.
    Inner(Npp<T>),
    /// A pointer to the list root.
    End(PinPtr<WaitList<T>>),
}

impl<T> LinkPtr<T> {
    /// Extract the pointer as a node-pointer. If the pointer is to the list
    /// root, returns `None` instead.
    fn as_node(&self) -> Option<Npp<T>> {
        if let Self::Inner(p) = self {
            Some(*p)
        } else {
            None
        }
    }

    /// Rewrites the "next" pointer in the object pointed to by `self` to point
    /// instead to `next`.
    ///
    /// # Safety
    ///
    /// This will dereference the pointer held in `self`, so all the usual rules
    /// about safe dereferencing of a pointer apply.
    ///
    /// In addition, to be used safely, this must not subvert the Link Valid
    /// Invariant on the overall WaitList/Node type family, which means that
    /// `next` must be a valid pointer and must point to either another node in
    /// the same list as the one pointed to by `self`, or the list root
    /// reachable from `self`.
    unsafe fn change_next(&self, next: Self) {
        fn replace_next<T>((prev, _next): (T, T), new: T) -> (T, T) {
            (prev, new)
        }

        unsafe {
            self.change(next, replace_next, replace_next);
        }
    }

    /// Rewrites the "prev" pointer in the object pointed to by `self` to point
    /// instead to `prev`.
    ///
    /// # Safety
    ///
    /// This will dereference the pointer held in `self`, so all the usual rules
    /// about safe dereferencing of a pointer apply.
    ///
    /// In addition, to be used safely, this must not subvert the Link Valid
    /// Invariant on the overall WaitList/Node type family, which means that
    /// `prev` must be a valid pointer and must point to either another node in
    /// the same list as the one pointed to by `self`, or the list root
    /// reachable from `self`.
    unsafe fn change_prev(&self, prev: Self) {
        fn replace_prev<T>((_prev, next): (T, T), new: T) -> (T, T) {
            (new, next)
        }

        unsafe {
            self.change(prev, replace_prev, replace_prev);
        }
    }

    /// Implementation factor of `change_{prev,next}` -- rewrite something
    /// through a `LinkPtr` generically.
    ///
    /// The signature of this function is kind of a hack, if we could take
    /// generic functions as higher-kinded we could only take one rewrite fn,
    /// but we can't.
    unsafe fn change(
        &self,
        neighbor: Self,
        rewrite1: impl FnOnce((Self, Self), Self) -> (Self, Self),
        rewrite2: impl FnOnce((Npp<T>, Npp<T>), Npp<T>) -> (Npp<T>, Npp<T>),
    ) {
        match self {
            Self::Inner(node) => {
                let node = unsafe { node.get() };
                let orig_links = node.links.get().unwrap();
                node.links.set(Some(rewrite1(orig_links, neighbor)));
            }
            Self::End(listptr) => {
                let list = unsafe { listptr.get() };
                match neighbor {
                    Self::End(other_list) => {
                        debug_assert!(*listptr == other_list);
                        list.links.set(None);
                    }
                    Self::Inner(node) => {
                        let list_heads = list.links.get()
                            .expect("list has become empty, but is about to be \
                                     made non-empty by detaching a node");
                        list.links.set(Some(rewrite2(list_heads, node)));
                    }
                }
            }
        }
    }
}

/// Force `Copy` whether or not `T` is `Copy`.
impl<T> Copy for LinkPtr<T> {}
/// Force `Clone` whether or not `T` is `Clone`.
impl<T> Clone for LinkPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
/// Implement equality independent of `T` since we just compare pointers.
/// Equality of `LinkPtr` is only used in integrity checks.
impl<T> PartialEq for LinkPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Inner(a), Self::Inner(b)) => a == b,
            (Self::End(a), Self::End(b)) => a == b,
            _ => false,
        }
    }
}
impl<T> Eq for LinkPtr<T> {}

/// Implement `Debug` independent of `T` since we only print addresses.
impl<T> Debug for LinkPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Inner(p) => f.debug_tuple("Inner").field(p).finish(),
            Self::End(p) => f.debug_tuple("End").field(p).finish(),
        }
    }
}

/// A `Waker` that does nothing. This is useful if you need a dummy `Waker` that
/// is independent of your choice of async runtimes, etc.
///
/// This is exposed because it's used under the hood by `create_node!`, and in
/// case you find other uses for it.
pub fn noop_waker() -> Waker {
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
        // clone
        |p| RawWaker::new(p, &NOOP_VTABLE),
        // wake
        |_| (),
        // wake_by_ref
        |_| (),
        // drop
        |_| (),
    );

    // Safety: the noop waker doesn't dereference the pointer we pass in here,
    // so we could hand in _literally anything_ without safety implications. But
    // the null pointer is nice for this.
    unsafe {
        Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::pin::pin;
    use core::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
    use core::task::Context;
    use std::sync::Arc;

    fn poll<T>(fut: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
        let waker = noop_waker();
        let mut ctx = Context::from_waker(&waker);
        fut.poll(&mut ctx)
    }

    /// Performs a list structural integrity check, panics if any issues are
    /// found.
    fn check<T: PartialOrd + Copy>(list: Pin<&WaitList<T>>) {
        let (tail, head) = if let Some((t, h)) = list.links.get() {
            (t, h)
        } else {
            // Empty list, nothing to check.
            return;
        };

        let mut cursor = head;
        let mut expected_prev = LinkPtr::End(PinPtr::new(list));
        let mut prev_value = None;
        loop {
            let node = unsafe { cursor.get() };
            let (prev, next) = node.links.get().expect("detached node in list");

            assert_eq!(prev, expected_prev, "corrupt node prev pointer");

            if let Some(pv) = prev_value {
                assert!(pv <= node.contents, "ascending order not maintained");
            }
            prev_value = Some(node.contents);

            match next {
                LinkPtr::Inner(next_node) => {
                    assert_ne!(next_node, cursor, "circular next link");
                    expected_prev = LinkPtr::Inner(cursor);
                    cursor = next_node;
                }
                LinkPtr::End(end_list) => {
                    assert_eq!(cursor, tail, "unexpected end node");
                    assert_eq!(end_list, PinPtr::new(list),
                        "cross-linked list");
                    break;
                }
            }
        }
    }

    #[allow(dead_code)] // useful when tests are failing
    fn dump<T>(list: Pin<&WaitList<T>>)
        where T: Debug,
    {
        println!("--- list dump ---");
        let mut index = 0;
        let mut candidate = list.links.get().map(|(_t, h)| h);
        while let Some(cptr) = candidate {
            // Safety: Link Valid Invariant
            let cref = unsafe { cptr.get() };

            println!("list index {index} @{cptr:?}: {cref:#?}");
            let next = cref.links.get().unwrap().1;
            candidate = next.as_node();
            index += 1;
        }
        println!("--- end of list dump ---");
    }

    fn spy_waker() -> (Waker, Arc<AtomicUsize>) {
        static SPY_VTABLE: RawWakerVTable = RawWakerVTable::new(
            // clone
            |arcptr| {
                unsafe { Arc::increment_strong_count(arcptr as *const AtomicUsize); }
                RawWaker::new(arcptr, &SPY_VTABLE)
            },
            // wake
            |arcptr| {
                let arc = unsafe { Arc::from_raw(arcptr as *const AtomicUsize) };
                arc.fetch_add(1, Ordering::Relaxed);
                drop(arc);
            },
            // wake_by_ref
            |arcptr| {
                let arc = unsafe { Arc::from_raw(arcptr as *const AtomicUsize) };

                arc.fetch_add(1, Ordering::Relaxed);

                // re-leak the arc so we can be reused.
                let _ = Arc::into_raw(arc);
            },
            // drop
            |arcptr| {
                let arc = unsafe { Arc::from_raw(arcptr as *const AtomicUsize) };
                drop(arc);
            },
            );

        let counter = Arc::new(AtomicUsize::new(0));
        let clone = Arc::clone(&counter);
        let w = unsafe {
            Waker::from_raw(RawWaker::new(
                Arc::into_raw(counter) as *const (),
                &SPY_VTABLE,
            ))
        };
        (w, clone)
    }

    #[test]
    fn hes_making_a_list() {
        create_list!(list, ());
        check(list.as_ref());
    }

    #[test]
    fn list_wake_oldest_empty() {
        create_list!(list, ());

        assert!(!list.wake_oldest());
    }

    #[test]
    fn list_wake_all_empty() {
        create_list!(list, ());

        list.wake_all();
    }

    #[test]
    fn insert_and_cancel_future() {
        create_list!(list, ());

        {
            let mut fut = pin!(list.wait(()));
            assert!(list.is_empty(), "wait should not eagerly attach");
            assert!(poll(fut.as_mut()).is_pending());
            assert!(!list.is_empty(), "wait should attach on first poll");
            check(list.as_ref());
        }
        assert!(list.is_empty(), "wait should detach on cancel");

        check(list.as_ref());
    }

    #[test]
    fn insert_two_and_cancel_out_of_order() {
        create_list!(list, ());

        {
            let mut fut2 = pin!(list.wait(()));
            {
                let mut fut1 = pin!(list.wait(()));

                assert!(poll(fut1.as_mut()).is_pending());
                assert!(poll(fut2.as_mut()).is_pending());
                check(list.as_ref());

                // fut1 dropped here:
            }
            check(list.as_ref());
            // fut2 dropped here:
        }
        assert!(list.is_empty());
        check(list.as_ref());
    }

    #[test]
    fn list_wake_while_node_inserted() {
        create_list!(list, ());
        {
            let (w, wake_count) = spy_waker();

            let mut wait1 = pin!(list.wait(()));
            let mut ctx = Context::from_waker(&w);

            // We can poll the insert future all we want but it doesn't resolve
            // while the node is in the list.
            assert_eq!(wait1.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wait1.as_mut().poll(&mut ctx), Poll::Pending);

            assert_eq!(wake_count.load(Ordering::Relaxed), 0);

            // Wake up one node (and check that effects happened)
            assert!(list.wake_oldest());

            assert_eq!(wake_count.load(Ordering::Relaxed), 1);

            // Now the future should resolve.
            assert_eq!(wait1.as_mut().poll(&mut ctx), Poll::Ready(()));
        }
    }

    #[test]
    fn insert_and_cancel_with_cleanup_action() {
        create_list!(list, ());

        // Flag we'll update from our cleanup action to detect that it's been
        // run.
        let future_dropped = AtomicBool::new(false);

        {
            // Insert with cleanup closure.
            let fut = pin!(list.wait_with_cleanup(
                (),
                || future_dropped.store(true, Ordering::Relaxed),
            ));

            // Future is currently in the "not yet attached" state.

            assert_eq!(future_dropped.load(Ordering::Relaxed), false,
                "Future must not run cleanup action until dropped");

            // Arrange to poll it.

            let (w, wake_count) = spy_waker();
            {
                let mut ctx = Context::from_waker(&w);

                assert_eq!(fut.poll(&mut ctx), Poll::Pending,
                    "Future should not resolve while node is attached");
            }

            // Wake the node, triggering our waker and detaching it from the
            // list.

            assert!(list.wake_oldest());
            assert_eq!(wake_count.load(Ordering::Relaxed), 1);

            // Drop the future without polling it.
        }
        assert_eq!(future_dropped.load(Ordering::Relaxed), true,
            "Future must run cleanup action when dropped \
             after detach without being polled");
    }

    #[test]
    fn test_insert_and_wait_not_eager() {
        create_list!(list);

        // Insertion must not happen eagerly, it must wait for the insert future to
        // be pinned and polled.
        {
            let _fut = pin!(list.wait(()));
            // Should not be able to wake it!
            assert_eq!(list.wake_oldest(), false);
            assert!(list.is_empty());
        }
        assert!(list.is_empty());
    }

    #[test]
    fn list_wake_le_ascending_order() {
        create_list!(list);
        {
            // Create a collection of four nodes with varying values. We expect
            // the list to maintain these in ascending order, regardless of the
            // order in which we insert them.
            let mut node1_fut = pin!(list.wait(1u32));
            let mut node2_fut = pin!(list.wait(2u32));
            let mut node3_fut = pin!(list.wait(3u32));
            let mut node4_fut = pin!(list.wait(4u32));

            assert!(list.is_empty());

            // Insert them in shuffled order.
            assert!(poll(node2_fut.as_mut()).is_pending());
            assert!(poll(node4_fut.as_mut()).is_pending());
            assert!(poll(node3_fut.as_mut()).is_pending());
            assert!(poll(node1_fut.as_mut()).is_pending());

            // Set up minimal async runtime state to poll them.
            let (w, wake_count) = spy_waker();
            let mut ctx = Context::from_waker(&w);

            // Verify our starting position:
            check(list.as_ref());
            assert_eq!(node1_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node2_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node3_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 0);

            // Start waking subsets. No nodes are <= 0:
            list.wake_le(0);
            check(list.as_ref());
            assert_eq!(node1_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node2_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node3_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 0);

            // One node is <= 1
            list.wake_le(1);
            check(list.as_ref());
            assert_eq!(node1_fut.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node2_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node3_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 1);

            // Two nodes are <= 3
            list.wake_le(3);
            check(list.as_ref());
            assert_eq!(node2_fut.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node3_fut.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node4_fut.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 3);

            // And one remaining node is <= 4
            list.wake_le(4);
            check(list.as_ref());
            assert_eq!(node4_fut.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(wake_count.load(Ordering::Relaxed), 4);
        }
    }

    #[test]
    fn list_wake_oldest() {
        create_list!(list);
        {
            // Make four nodes.
            let mut node1 = pin!(list.wait(()));
            let mut node2 = pin!(list.wait(()));
            let mut node3 = pin!(list.wait(()));
            let mut node4 = pin!(list.wait(()));

            // Poll to insert them in order, so node1 is oldest.
            assert!(poll(node1.as_mut()).is_pending());
            assert!(poll(node2.as_mut()).is_pending());
            assert!(poll(node3.as_mut()).is_pending());
            assert!(poll(node4.as_mut()).is_pending());

            // Set up minimal async runtime state to poll them.
            let (w, wake_count) = spy_waker();
            let mut ctx = Context::from_waker(&w);

            // Verify our starting position:
            check(list.as_ref());
            assert_eq!(node1.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node2.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node3.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 0);

            // Start waking individual nodes.
            assert!(list.wake_oldest());
            check(list.as_ref());
            assert_eq!(node1.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node2.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node3.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 1);

            assert!(list.wake_oldest());
            check(list.as_ref());
            assert_eq!(node2.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node3.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(node4.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 2);

            assert!(list.wake_oldest());
            check(list.as_ref());
            assert_eq!(node3.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(node4.as_mut().poll(&mut ctx), Poll::Pending);
            assert_eq!(wake_count.load(Ordering::Relaxed), 3);

            assert!(list.wake_oldest());
            check(list.as_ref());
            assert_eq!(node4.as_mut().poll(&mut ctx), Poll::Ready(()));
            assert_eq!(wake_count.load(Ordering::Relaxed), 4);
        }
    }
}
