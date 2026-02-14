use candle_core::{Tensor, Result, IndexOp};
use std::thread;
use std::sync::Arc;
use crate::block::min_lstm::{MinLstm, MinLstmConfig};

pub struct MinLstmThreaded {
    inner: Arc<MinLstm>,
    _config: MinLstmConfig,
    num_threads: usize,
}

impl MinLstmThreaded {
    /// Crea una nueva instancia de MinLstm que utiliza hilos para el procesamiento por batch.
    /// `num_threads` especifica el máximo de hilos simultáneos a utilizar.
    pub fn new(inner: MinLstm, config: MinLstmConfig, num_threads: usize) -> Self {
        Self { 
            inner: Arc::new(inner), 
            _config: config,
            num_threads,
        }
    }

    /// Forward que paraleliza el procesamiento por Batch usando hilos nativos.
    pub fn forward(&self, x: &Tensor, _prev_h: Option<&Tensor>, return_next: bool) -> Result<(Tensor, Option<Tensor>)> {
        let (b, s, d) = x.dims3()?;
        
        // El número de hilos real será el mínimo entre el batch y lo configurado
        let actual_threads = std::cmp::min(b, self.num_threads);
        
        // Si el batch es pequeño o solo hay un hilo configurado, usamos el forward normal
        if b <= 1 || actual_threads <= 1 {
            return self.inner.forward(x, _prev_h, return_next);
        }

        let batch_per_thread = (b + actual_threads - 1) / actual_threads;
        let mut handles = vec![];

        for t_idx in 0..actual_threads {
            let start_b = t_idx * batch_per_thread;
            let end_b = std::cmp::min(start_b + batch_per_thread, b);
            if start_b >= end_b { break; }

            let x_chunk = x.narrow(0, start_b, end_b - start_b)?;
            // Explicitly annotate type to help inference
            let model_ref: Arc<MinLstm> = Arc::clone(&self.inner);
            
            handles.push(thread::spawn(move || {
                model_ref.forward(&x_chunk, None, false)
            }));
        }

        let mut results = Vec::with_capacity(actual_threads);
        for handle in handles {
            let (out, _) = handle.join().map_err(|_| candle_core::Error::Msg("Thread panic".into()))??;
            results.push(out);
        }

        let combined_h = Tensor::cat(&results, 0)?;
        
        let last_state = if return_next {
            Some(combined_h.i((.., s-1, ..))?.contiguous()?)
        } else {
            None
        };

        Ok((combined_h, last_state))
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}
