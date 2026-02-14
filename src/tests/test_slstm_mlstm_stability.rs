use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::error::Error;
use xlstm::{MLstm, MLstmconfig, SLstm, SLstmconfig};

fn count_nan_inf(t: &Tensor) -> Result<(usize, usize), Box<dyn Error>> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let mut nan = 0usize;
    let mut inf = 0usize;
    for x in v {
        if x.is_nan() {
            nan += 1;
        }
        if x.is_infinite() {
            inf += 1;
        }
    }
    Ok((nan, inf))
}

struct ReferenceLstm {
    w_ih: Tensor,
    w_hh: Tensor,
    b: Tensor,
    hidden_size: usize,
}

impl ReferenceLstm {
    fn new(device: &Device, input_size: usize, hidden_size: usize) -> Result<Self, Box<dyn Error>> {
        let w_ih = Tensor::randn(0.0f32, 0.02f32, (input_size, 4 * hidden_size), device)?;
        let w_hh = Tensor::randn(0.0f32, 0.02f32, (hidden_size, 4 * hidden_size), device)?;
        let b = Tensor::zeros((4 * hidden_size,), DType::F32, device)?;
        Ok(Self {
            w_ih,
            w_hh,
            b,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let mut h = Tensor::zeros((batch, self.hidden_size), DType::F32, x.device())?;
        let mut c = Tensor::zeros((batch, self.hidden_size), DType::F32, x.device())?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq);

        for t in 0..seq {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
            let b = self
                .b
                .unsqueeze(0)?
                .broadcast_as((batch, 4 * self.hidden_size))?;
            let gates = x_t
                .matmul(&self.w_ih)?
                .add(&h.matmul(&self.w_hh)?)?
                .add(&b)?;
            let chunks = gates.chunk(4, 1)?;
            let i = candle_nn::ops::sigmoid(&chunks[0])?;
            let f = candle_nn::ops::sigmoid(&chunks[1])?;
            let g = chunks[2].tanh()?;
            let o = candle_nn::ops::sigmoid(&chunks[3])?;
            c = f.mul(&c)?.add(&i.mul(&g)?)?;
            h = o.mul(&c.tanh()?)?;
            outputs.push(h.unsqueeze(1)?);
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }
}

fn test_forward_stats(name: &str, y: &Tensor) -> Result<(), Box<dyn Error>> {
    let (nan, inf) = count_nan_inf(y)?;
    println!(
        "{} - shape: {:?}, mean: {:.4}, min: {:.4}, max: {:.4}, NaN: {}, Inf: {}",
        name,
        y.dims(),
        y.mean_all()?.to_scalar::<f32>()?,
        y.min_all()?.to_scalar::<f32>()?,
        y.max_all()?.to_scalar::<f32>()?,
        nan,
        inf
    );
    Ok(())
}

fn test_slstm(device: &Device) -> Result<(), Box<dyn Error>> {
    let d_input = 64;
    let d_hidden = 64;
    let num_layers = 1;

    let config = SLstmconfig::new(d_input, d_hidden, num_layers);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let slstm: SLstm = config.init(vb)?;

    let x = Tensor::randn(0.0f32, 1.0f32, (16, 32, d_input), device)?;
    let (y, _state) = slstm.forward(&x, None)?;
    test_forward_stats("sLSTM", &y)?;

    let ref_lstm = ReferenceLstm::new(device, d_input, d_hidden)?;
    let y_ref = ref_lstm.forward(&x)?;
    test_forward_stats("LSTM ref", &y_ref)?;

    Ok(())
}

fn test_mlstm(device: &Device) -> Result<(), Box<dyn Error>> {
    let d_input = 64;
    let d_hidden = 64;
    let num_layers = 1;
    let num_heads = 4;

    let config = MLstmconfig::new(d_input, d_hidden, num_layers, num_heads);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mlstm: MLstm = config.init(vb)?;

    let x = Tensor::randn(0.0f32, 1.0f32, (16, 32, d_input), device)?;
    let (y, _state) = mlstm.forward(&x, None)?;
    test_forward_stats("mLSTM", &y)?;

    let ref_lstm = ReferenceLstm::new(device, d_input, d_hidden)?;
    let y_ref = ref_lstm.forward(&x)?;
    test_forward_stats("LSTM ref", &y_ref)?;

    Ok(())
}

fn train_cumsum_slstm_vs_mlstm(device: &Device) -> Result<(), Box<dyn Error>> {
    use candle_nn::optim::AdamW;

    let batch_size = 32;
    let seq_len = 30;
    let dim = 64;
    let steps = 100;

    let u0 = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, dim), device)?;
    let scale = Tensor::full(0.1f32, u0.shape(), device)?;
    let u = u0.mul(&scale)?;
    let target = u.cumsum(1)?;

    let slstm_cfg = SLstmconfig::new(dim, dim, 1);
    let slstm_vm = VarMap::new();
    let slstm_vb = VarBuilder::from_varmap(&slstm_vm, DType::F32, device);
    let slstm = slstm_cfg.init(slstm_vb)?;
    let slstm_params = {
        let data = slstm_vm.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut opt_slstm = AdamW::new_lr(slstm_params, 8e-4)?;

    let mlstm_cfg = MLstmconfig::new(dim, dim, 1, 4);
    let mlstm_vm = VarMap::new();
    let mlstm_vb = VarBuilder::from_varmap(&mlstm_vm, DType::F32, device);
    let mlstm = mlstm_cfg.init(mlstm_vb)?;
    let mlstm_params = {
        let data = mlstm_vm.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut opt_mlstm = AdamW::new_lr(mlstm_params, 8e-4)?;

    let mut slstm_first: Option<f32> = None;
    let mut mlstm_first: Option<f32> = None;
    let mut slstm_last = 0.0f32;
    let mut mlstm_last = 0.0f32;

    for _ in 0..steps {
        let out_s = slstm.forward(&u, None)?.0;
        let loss_s = (out_s.sub(&target)?).sqr()?.mean_all()?;
        slstm_last = loss_s.to_scalar::<f32>()?;
        if slstm_first.is_none() {
            slstm_first = Some(slstm_last);
        }
        let grads_s = loss_s.backward()?;
        opt_slstm.step(&grads_s)?;

        let out_m = mlstm.forward(&u, None)?.0;
        let loss_m = (out_m.sub(&target)?).sqr()?.mean_all()?;
        mlstm_last = loss_m.to_scalar::<f32>()?;
        if mlstm_first.is_none() {
            mlstm_first = Some(mlstm_last);
        }
        let grads_m = loss_m.backward()?;
        opt_mlstm.step(&grads_m)?;
    }

    if let (Some(a), Some(b)) = (slstm_first, mlstm_first) {
        println!("sLSTM cumsum loss inicio: {:.6} -> fin: {:.6}", a, slstm_last);
        println!("mLSTM cumsum loss inicio: {:.6} -> fin: {:.6}", b, mlstm_last);
    }
    println!("Puntaje sLSTM cumsum (loss final): {:.6}", slstm_last);
    println!("Puntaje mLSTM cumsum (loss final): {:.6}", mlstm_last);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;

    println!("=== Test Estabilidad sLSTM / mLSTM ===");

    println!("\n--- Test 1: sLSTM forward + comparación con LSTM normal ---");
    test_slstm(&device)?;

    println!("\n--- Test 2: mLSTM forward + comparación con LSTM normal ---");
    test_mlstm(&device)?;

    println!("\n--- Test 3: Entrenamiento tarea secuencial (suma acumulada) sLSTM vs mLSTM ---");
    train_cumsum_slstm_vs_mlstm(&device)?;

    println!("\n=== Fin ===");
    Ok(())
}
