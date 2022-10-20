//! THIS WILL ONLY RUN UNDER MIRI.
//!
//! Copy of one of the library test cases, running in `no_std` mode under Miri.
//! This helps to validate that the `no_std` code works.
//!
//! To use: `cargo +nightly miri run --bin no_std`

#![feature(start, core_intrinsics, lang_items)]
#![no_std]

use core::fmt::Write;
use core::future::Future;
use core::task::{Context, Poll};
use lilist::{create_list, noop_waker, create_node};

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    create_list!(list, ());
    writeln!(HostOut, "running wake test...").ok();
    {
        let w = noop_waker();
        create_node!(node, (), noop_waker());

        let node1_wait = list.insert_and_wait(node);
        pin_utils::pin_mut!(node1_wait);
        let mut ctx = Context::from_waker(&w);

        // We can poll the insert future all we want but it doesn't resolve
        // while the node is in the list.
        assert_eq!(node1_wait.as_mut().poll(&mut ctx), Poll::Pending);
        assert_eq!(node1_wait.as_mut().poll(&mut ctx), Poll::Pending);

        // Wake up one node (and check that effects happened)
        assert!(list.wake_oldest());

        // Now the future should resolve.
        assert_eq!(node1_wait.as_mut().poll(&mut ctx), Poll::Ready(()));
    }

    writeln!(HostOut, "tests passed").ok();
    0
}

extern "Rust" {
    fn miri_write_to_stdout(bytes: &[u8]);
    fn miri_write_to_stderr(bytes: &[u8]);
}

struct HostOut;

impl Write for HostOut {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe {
            miri_write_to_stdout(s.as_bytes());
        }
        Ok(())
    }
}

struct HostErr;

impl Write for HostErr {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe {
            miri_write_to_stderr(s.as_bytes());
        }
        Ok(())
    }
}

#[cfg(panic = "abort")]
#[panic_handler]
fn panic_handler(panic_info: &core::panic::PanicInfo) -> ! {
    writeln!(HostErr, "{panic_info}").ok();
    core::intrinsics::abort();
}

#[cfg(panic = "abort")]
#[lang = "eh_personality"]
fn eh_personality() {}
