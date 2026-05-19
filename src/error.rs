use std::io;

#[derive(thiserror::Error, Debug)]
pub enum Ds4Error {
    #[error("{0}")]
    InvalidArgument(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("prompt exceeds context: prompt={prompt_len} ctx={ctx_size}")]
    ContextExceeded { prompt_len: usize, ctx_size: usize },
    #[error("{0}")]
    Protocol(String),
    #[error("{0}")]
    Unavailable(String),
}

pub type Result<T> = std::result::Result<T, Ds4Error>;
