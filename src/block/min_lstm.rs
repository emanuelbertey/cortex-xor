use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder, Linear};
use candle_nn::init::Init;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinLstmConfig {
    pub dim: usize,
    pub expansion_factor: f32,
    pub heads: usize,
    // Variables de ajuste de estabilidad
    pub weight_stdev: f64,
    pub forget_bias: f32,
    pub epsilon: f32,
    pub log_clamp: f32,
    pub scan_clamp: f32,
}

impl MinLstmConfig {
    pub fn new(dim: usize) -> Self {
        Self { 
            dim, 
            expansion_factor: 1.0, 
            heads: 1,
            // Valores por defecto sugeridos
            weight_stdev: 0.03,
            forget_bias: 2.2,
            epsilon: 1e-6,
            log_clamp: 1e-4,
            scan_clamp: 12.0,
        }
    }

    pub fn with_expansion_factor(mut self, factor: f32) -> Self {
        self.expansion_factor = factor;
        self
    }

    pub fn with_stability(mut self, stdev: f64, f_bias: f32, scan: f32) -> Self {
        self.weight_stdev = stdev;
        self.forget_bias = f_bias;
        self.scan_clamp = scan;
        self
    }

    pub fn init(&self, vb: VarBuilder) -> Result<MinLstm> {
        let d_inner = (self.dim as f32 * self.expansion_factor) as usize;
        
        let init_ws = Init::Randn { 
            mean: 0.0, 
            stdev: self.weight_stdev 
        };
        
        let w = vb.get_with_hints((d_inner * 3, self.dim), "weight", init_ws)?;
        
        let mut b_v = vec![0.0f32; d_inner * 3];
        for i in 0..d_inner {
            b_v[i + d_inner] = self.forget_bias; 
        }
        let b = vb.get_with_hints(d_inner * 3, "bias", Init::Const(0.0))?;
        let bias_offset = Tensor::from_vec(b_v, (1, 1, d_inner * 3), w.device())?;

        Ok(MinLstm { 
            to_hfg: Linear::new(w, Some(b)), 
            bias_offset,
            dim_inner: d_inner,
            config: self.clone(),
        })
    }
}

#[derive(Debug)]
pub struct MinLstm {
    to_hfg: Linear,
    bias_offset: Tensor,
    dim_inner: usize,
    config: MinLstmConfig,
}

impl MinLstm {
    pub fn forward(&self, x: &Tensor, prev_h: Option<&Tensor>, return_next: bool) -> Result<(Tensor, Option<Tensor>)> {
        let (b, seq, _) = x.dims3()?;
        let dev = x.device();
        let dtype = x.dtype();

        let hfg = self.to_hfg.forward(x)?.broadcast_add(&self.bias_offset)?; 
        // Escalado 1/sqrt(d) como en el paper
        let inv_sqrt = 1.0f64 / (self.dim_inner as f64).sqrt();
        let hfg_scaled = hfg.affine(inv_sqrt, 0.0)?;
        // Gates del paper: g con tanh, f e i con sigmoide
        let g = hfg_scaled.narrow(2, 0, self.dim_inner)?.tanh()?;
        let f = candle_nn::ops::sigmoid(&hfg_scaled.narrow(2, self.dim_inner, self.dim_inner)?)?;
        let i = candle_nn::ops::sigmoid(&hfg_scaled.narrow(2, 2 * self.dim_inner, self.dim_inner)?)?;

        let eps_t = Tensor::new(self.config.epsilon, dev)?.to_dtype(dtype)?;
        let den = f.broadcast_add(&i)?.broadcast_add(&eps_t)?;
        let f_hat = f.div(&den)?;
        let i_hat = i.div(&den)?;

        // Scan simple del paper: a_t = exp(cumsum(log f_hat)), b_t = (i*g)/a_t, h_t = cumsum(b_t) * a_t
        let log_f = f_hat.log()?;
        let a_star = log_f.cumsum(1)?;
        let a_t_f64 = a_star.to_dtype(DType::F64)?.exp()?;
        let ig_f64 = i_hat.mul(&g)?.to_dtype(DType::F64)?;
        let b_t_f64 = ig_f64.div(&a_t_f64)?;
        let h_t_parallel_f64 = b_t_f64.cumsum(1)?.mul(&a_t_f64)?;
        let h_t_parallel = h_t_parallel_f64.to_dtype(dtype)?;
        let h_t = if let Some(h0) = prev_h {
            let h0_decayed = h0
                .reshape((b, 1, self.dim_inner))?
                .to_dtype(DType::F64)?
                .mul(&a_t_f64)?
                .to_dtype(dtype)?;
            h_t_parallel.add(&h0_decayed)?
        } else {
            h_t_parallel
        }.contiguous()?;

        let next_h = if return_next {
            Some(h_t.narrow(1, seq - 1, 1)?.contiguous()?)
        } else {
            None
        };

        Ok((h_t, next_h))
    }
}
