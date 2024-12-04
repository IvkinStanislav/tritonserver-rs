use std::{collections::HashMap, future::Future};

use crate::{
    error::{Error, ErrorCode},
    memory::Buffer,
    request::infer::*,
    Response,
};

/// Awaiting on this structure will returt result of the inference: Ok([Response]) or Err([InferenceError]).
impl Future for ResponseFuture {
    type Output = Result<Response, InferenceError>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.input_release.is_some() {
            log::debug!("ResponseFuture has unhandled InputRelease. \
                Ignore this message if there is no need to handle returned input resources. They will be dropped.
            ");
        }
        unsafe { self.map_unchecked_mut(|this| &mut this.response_receiver) }
            .poll(cx)
            .map(|recv_res| match recv_res {
                Ok(res) => res,
                Err(recv_err) => Err(Error::new(
                    ErrorCode::Internal,
                    format!("response receive error: {recv_err}"),
                )
                .into()),
            })
    }
}

impl ResponseFuture {
    /// Blocking await to call outside of asynchronous contexts.
    ///
    /// # Panics
    ///
    /// This function panics if called within an asynchronous execution context.
    pub fn blocking_recv(self) -> Result<Response, InferenceError> {
        match self.response_receiver.blocking_recv() {
            Ok(res) => res,
            Err(recv_err) => Err(Error::new(
                ErrorCode::Internal,
                format!("response receive error: {recv_err}"),
            )
            .into()),
        }
    }

    /// Get the future to return the input buffers assigned to the Request.
    ///
    /// **NOTE**: this function should be called at most once. Otherwise it will return garbage. \
    /// **Note** that input buffer can be released in any time from the start of the inference
    /// to the end of it.
    pub fn get_input_release(&mut self) -> InputRelease {
        self.input_release.take().unwrap_or_else(|| {
            log::error!("ResponseFuture::get_input_release was invoked twice in a row. Empty future is returned");
            let (_, rx) = tokio::sync::oneshot::channel();
            InputRelease(rx)
        })
    }
}

/// Awaiting on input buffers returnal from the inference.
///
/// Note that input buffer can be released in any time from the start of the inference
/// to the end of it.
impl Future for InputRelease {
    type Output = Result<HashMap<String, Buffer>, Error>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        unsafe { self.map_unchecked_mut(|this| &mut this.0) }
            .poll(cx)
            .map_err(|recv_err| {
                Error::new(
                    ErrorCode::Internal,
                    format!("Receive input buffer error: {recv_err}"),
                )
            })
    }
}

impl InputRelease {
    /// Blocking receive to call outside of asynchronous contexts.\
    /// # Panics
    /// This function panics if called within an asynchronous execution context.
    pub fn blocking_recv(self) -> Result<HashMap<String, Buffer>, Error> {
        self.0.blocking_recv().map_err(|recv_error| {
            Error::new(
                ErrorCode::Internal,
                format!("Receive input buffer error: {recv_error}"),
            )
        })
    }
}
