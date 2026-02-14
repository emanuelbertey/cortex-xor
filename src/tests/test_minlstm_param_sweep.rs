use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use xlstm::MinLstmConfig;
use std::env;

fn make_data(batch: usize, seq: usize, dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::randn(0.0, 1.0, (batch, seq, dim), device)?.to_dtype(DType::F32)?;
    let x_mean = x.mean_keepdim(D::Minus1)?;
    let y = x_mean.cumsum(1)?;
    Ok((x, y.squeeze(D::Minus1)?))
}

fn make_parity_data(batch: usize, seq: usize, dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let raw = Tensor::randn(0.0, 1.0, (batch, seq, 1), device)?.to_dtype(DType::F32)?;
    let x_bits = raw.ge(0.0)?.to_dtype(DType::F32)?;
    let x = x_bits.broadcast_as((batch, seq, dim))?;
    let cumsum = x_bits.squeeze(D::Minus1)?.cumsum(1)?;
    let two = Tensor::new(2.0f32, device)?;
    let div_two = cumsum.broadcast_div(&two)?;
    let floor_div_two = div_two.floor()?;
    let y = cumsum.broadcast_sub(&floor_div_two.broadcast_mul(&two)?)?;
    Ok((x, y))
}

fn print_stats(t: &Tensor, name: &str) -> Result<()> {
    let m = t.mean_all()?.to_scalar::<f32>()?;
    let mx = t.max_all()?.to_scalar::<f32>()?;
    let mn = t.min_all()?.to_scalar::<f32>()?;
    println!("{} - Mean: {:.4}, Min: {:.4}, Max: {:.4}", name, m, mn, mx);
    Ok(())
}

fn run_stability(cfg: &MinLstmConfig, device: &Device) -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = cfg.clone().init(vb)?;
    let x = Tensor::randn(0.0, 1.0, (16, 32, cfg.dim), device)?.to_dtype(DType::F32)?;
    let (out, _) = model.forward(&x, None, false)?;
    print_stats(&x, "Input")?;
    print_stats(&out, "Output")?;
    Ok(())
}

fn train_once(cfg: &MinLstmConfig, steps: usize, device: &Device) -> Result<((f32, f32), (f32, f32))> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = cfg.init(vb)?;
    
    // AQUÍ ESTÁ EL OPTIMIZADOR CON EL LR CORRECTO (3e-3)
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 0.003, 
        ..Default::default()
    })?;

    let (x_orig, y_orig) = make_data(16, 32, cfg.dim, device)?;
    let (x_par, y_par) = make_parity_data(16, 32, cfg.dim, device)?;

    let out0 = model.forward(&x_orig, None, false)?.0;
    let l0_orig = candle_nn::loss::mse(&out0.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_orig)?.to_scalar::<f32>()?;

    let out0p = model.forward(&x_par, None, false)?.0;
    let l0_par = candle_nn::loss::mse(&out0p.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_par)?.to_scalar::<f32>()?;

    for _ in 0..steps {
        let (out, _) = model.forward(&x_orig, None, false)?;
        opt.backward_step(&candle_nn::loss::mse(&out.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_orig)?)?;

        let (outp, _) = model.forward(&x_par, None, false)?;
        opt.backward_step(&candle_nn::loss::mse(&outp.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_par)?)?;
    }

    let outf = model.forward(&x_orig, None, false)?.0;
    let lf_orig = candle_nn::loss::mse(&outf.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_orig)?.to_scalar::<f32>()?;

    let outfp = model.forward(&x_par, None, false)?.0;
    let lf_par = candle_nn::loss::mse(&outfp.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?, &y_par)?.to_scalar::<f32>()?;

    Ok(((l0_orig, lf_orig), (l0_par, lf_par)))
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let base = MinLstmConfig::new(256);
    let args: Vec<String> = env::args().collect();
    let max_cases: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let train_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    println!("=== Barrido: Test Original + Paridad (LR=3e-3, Steps={}) ===", train_steps);

    for i in 0..max_cases {
        let t = if max_cases > 1 { i as f32 / (max_cases as f32 - 1.0) } else { 0.0 };
        let mut cfg = base.clone()
            .with_expansion_factor(1.0 + t)
            .with_stability(0.01 + (t as f64)*0.04, 0.5 + t*1.5, 5.0 + t*7.0);
        
        println!("--- Caso {} (fb={:.2}, sc={:.1}) ---", i + 1, cfg.forget_bias, cfg.scan_clamp);
        run_stability(&cfg, &device)?;

        match train_once(&cfg, train_steps, &device) {
            Ok((orig, par)) => {
                println!("   [RESULTADO ORIGINAL] Loss: {:.6} -> {:.6}", orig.0, orig.1);
                println!("   [RESULTADO PARIDAD ] Loss: {:.6} -> {:.6}", par.0, par.1);
            },
            Err(e) => println!("   [!] Error: {:?}", e),
        }
        println!();
    }
    Ok(())
}