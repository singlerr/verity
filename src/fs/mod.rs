pub mod diff;
pub mod read;
pub mod walk;
pub mod write;

pub use diff::{compute_diff, render_diff, DiffLine};
pub use read::read_file;
pub use walk::{walk_dir, FileTree};
pub use write::write_file;
