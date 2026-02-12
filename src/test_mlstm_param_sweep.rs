/*!
    Test de estrés y barrido de parámetros para mLSTM.
    Barre: weight_stdev, forget_bias, input_gate_bias, epsilon, exp_clamp,
           norm_clamp_min/max, exp_gate_scale, use_separate_bias.
    Reporta estabilidad numérica y convergencia de entrenamiento.
*/

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use xlstm::MLstmconfig;
use std::env;

// ─── Datos sintéticos ────────────────────────────────────────────────

/// Regresión: y = cumsum(mean(x, dim=-1))
fn make_data(batch: usize, seq: usize, dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::randn(0.0, 1.0, (batch, seq, dim), device)?.to_dtype(DType::F32)?;
    let x_mean = x.mean_keepdim(D::Minus1)?;
    let y = x_mean.cumsum(1)?;
    Ok((x, y.squeeze(D::Minus1)?))
}

/// Paridad: y_t = cumsum(bits) mod 2
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

// ─── Utilidades ──────────────────────────────────────────────────────

fn check_nan_inf(t: &Tensor) -> Result<(bool, bool)> {
    let m = t.mean_all()?.to_scalar::<f32>()?;
    let mx = t.max_all()?.to_scalar::<f32>()?;
    let mn = t.min_all()?.to_scalar::<f32>()?;
    let has_nan = m.is_nan() || mx.is_nan() || mn.is_nan();
    let has_inf = mx.is_infinite() || mn.is_infinite();
    Ok((has_nan, has_inf))
}

fn print_stats(t: &Tensor, name: &str) -> Result<()> {
    let m = t.mean_all()?.to_scalar::<f32>()?;
    let mx = t.max_all()?.to_scalar::<f32>()?;
    let mn = t.min_all()?.to_scalar::<f32>()?;
    let (has_nan, has_inf) = check_nan_inf(t)?;
    println!(
        "   {} | Mean: {:>10.4} | Min: {:>10.4} | Max: {:>10.4} | NaN:{} Inf:{}",
        name, m, mn, mx,
        if has_nan { " ⚠️" } else { " ✓" },
        if has_inf { " ⚠️" } else { " ✓" },
    );
    Ok(())
}

// ─── Test de estabilidad numérica (forward sin entrenamiento) ────────

fn run_stability(cfg: &MLstmconfig, device: &Device) -> Result<bool> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = cfg.init(vb)?;
    let x = Tensor::randn(0.0, 1.0, (8, 64, cfg.d_input), device)?.to_dtype(DType::F32)?;
    let (out, _states) = model.forward(&x, None)?;

    let (has_nan, has_inf) = check_nan_inf(&out)?;

    print_stats(&x, "Input ")?;
    print_stats(&out, "Output")?;

    Ok(!has_nan && !has_inf)
}

// ─── Entrenamiento corto para medir convergencia ────────────────────

fn train_once(cfg: &MLstmconfig, steps: usize, device: &Device) -> Result<((f32, f32), (f32, f32))> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = cfg.init(vb)?;

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 0.003,
        ..Default::default()
    })?;

    let _dim = cfg.d_hidden;
    let (x_orig, y_orig) = make_data(8, 32, cfg.d_input, device)?;
    let (x_par, y_par)   = make_parity_data(8, 32, cfg.d_input, device)?;

    // ── Loss inicial ──
    let out0 = model.forward(&x_orig, None)?.0;
    let out0_reduced = out0.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
    let l0_orig = candle_nn::loss::mse(&out0_reduced, &y_orig)?.to_scalar::<f32>()?;

    let out0p = model.forward(&x_par, None)?.0;
    let out0p_reduced = out0p.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
    let l0_par = candle_nn::loss::mse(&out0p_reduced, &y_par)?.to_scalar::<f32>()?;

    // ── Entrenamiento ──
    for step in 0..steps {
        // Tarea original
        let (out, _) = model.forward(&x_orig, None)?;
        let out_r = out.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
        let loss = candle_nn::loss::mse(&out_r, &y_orig)?;

        // Verificar NaN temprano
        let lv = loss.to_scalar::<f32>()?;
        if lv.is_nan() || lv.is_infinite() {
            println!("      ⚠️  NaN/Inf en step {} (tarea original), abortando", step);
            return Ok(((l0_orig, f32::NAN), (l0_par, f32::NAN)));
        }
        opt.backward_step(&loss)?;

        // Tarea paridad
        let (outp, _) = model.forward(&x_par, None)?;
        let outp_r = outp.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
        let lossp = candle_nn::loss::mse(&outp_r, &y_par)?;
        opt.backward_step(&lossp)?;
    }

    // ── Loss final ──
    let outf = model.forward(&x_orig, None)?.0;
    let outf_r = outf.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
    let lf_orig = candle_nn::loss::mse(&outf_r, &y_orig)?.to_scalar::<f32>()?;

    let outfp = model.forward(&x_par, None)?.0;
    let outfp_r = outfp.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
    let lf_par = candle_nn::loss::mse(&outfp_r, &y_par)?.to_scalar::<f32>()?;

    Ok(((l0_orig, lf_orig), (l0_par, lf_par)))
}

// ─── Definición de casos de barrido ─────────────────────────────────

struct SweepCase {
    label: String,
    cfg: MLstmconfig,
}

fn build_sweep_cases(dim: usize, heads: usize) -> Vec<SweepCase> {
    let base = || MLstmconfig::new(dim, dim, 1, heads);
    vec![
        // ─── 1. Baseline (defaults) ───
        SweepCase {
            label: "BASELINE (defaults)".into(),
            cfg: base(),
        },

        // ─── 2. weight_stdev sweep ───
        SweepCase {
            label: "weight_stdev=0.005 (bajo)".into(),
            cfg: { let mut c = base(); c.weight_stdev = 0.005; c },
        },
        SweepCase {
            label: "weight_stdev=0.1 (alto)".into(),
            cfg: { let mut c = base(); c.weight_stdev = 0.1; c },
        },
        SweepCase {
            label: "weight_stdev=0.5 (extremo)".into(),
            cfg: { let mut c = base(); c.weight_stdev = 0.5; c },
        },

        // ─── 3. forget_bias sweep ───
        SweepCase {
            label: "forget_bias=0.0 (sin retención)".into(),
            cfg: { let mut c = base(); c.forget_bias = 0.0; c },
        },
        SweepCase {
            label: "forget_bias=1.0 (retención media)".into(),
            cfg: { let mut c = base(); c.forget_bias = 1.0; c },
        },
        SweepCase {
            label: "forget_bias=3.0 (retención fuerte)".into(),
            cfg: { let mut c = base(); c.forget_bias = 3.0; c },
        },

        // ─── 4. input_gate_bias sweep ───
        SweepCase {
            label: "input_gate_bias=-1.0 (escritura lenta)".into(),
            cfg: { let mut c = base(); c.input_gate_bias = -1.0; c },
        },
        SweepCase {
            label: "input_gate_bias=0.5 (escritura rápida)".into(),
            cfg: { let mut c = base(); c.input_gate_bias = 0.5; c },
        },
        SweepCase {
            label: "input_gate_bias=2.0 (escritura agresiva)".into(),
            cfg: { let mut c = base(); c.input_gate_bias = 2.0; c },
        },

        // ─── 5. exp_gate_scale sweep ───
        SweepCase {
            label: "exp_gate_scale=1.0 (sin escalar)".into(),
            cfg: { let mut c = base(); c.exp_gate_scale = 1.0; c },
        },
        SweepCase {
            label: "exp_gate_scale=4.0 (conservador)".into(),
            cfg: { let mut c = base(); c.exp_gate_scale = 4.0; c },
        },
        SweepCase {
            label: "exp_gate_scale=8.0 (muy conservador)".into(),
            cfg: { let mut c = base(); c.exp_gate_scale = 8.0; c },
        },

        // ─── 6. exp_clamp sweep ───
        SweepCase {
            label: "exp_clamp=5.0 (restringido)".into(),
            cfg: { let mut c = base(); c.exp_clamp = 5.0; c },
        },
        SweepCase {
            label: "exp_clamp=50.0 (amplio)".into(),
            cfg: { let mut c = base(); c.exp_clamp = 50.0; c },
        },

        // ─── 7. epsilon sweep ───
        SweepCase {
            label: "epsilon=1e-8 (menos tolerancia)".into(),
            cfg: { let mut c = base(); c.epsilon = 1e-8; c },
        },
        SweepCase {
            label: "epsilon=1e-3 (más tolerancia)".into(),
            cfg: { let mut c = base(); c.epsilon = 1e-3; c },
        },

        // ─── 8. norm_clamp sweep ───
        SweepCase {
            label: "norm_clamp=[1e-8, 1e6] (estricto)".into(),
            cfg: { let mut c = base(); c.norm_clamp_min = 1e-8; c.norm_clamp_max = 1e6; c },
        },
        SweepCase {
            label: "norm_clamp=[1e-3, 1e3] (relajado)".into(),
            cfg: { let mut c = base(); c.norm_clamp_min = 1e-3; c.norm_clamp_max = 1e3; c },
        },

        // ─── 9. use_separate_bias = false ───
        SweepCase {
            label: "use_separate_bias=false (bias uniforme)".into(),
            cfg: { let mut c = base(); c.use_separate_bias = false; c },
        },

        // ─── 10. Combinaciones extremas ───
        SweepCase {
            label: "COMBO: agresivo (ws=0.1, fb=0.0, igb=1.0, egs=1.0)".into(),
            cfg: {
                let mut c = base();
                c.weight_stdev = 0.1;
                c.forget_bias = 0.0;
                c.input_gate_bias = 1.0;
                c.exp_gate_scale = 1.0;
                c
            },
        },
        SweepCase {
            label: "COMBO: conservador (ws=0.005, fb=2.0, igb=-0.5, egs=6.0, ec=8.0)".into(),
            cfg: {
                let mut c = base();
                c.weight_stdev = 0.005;
                c.forget_bias = 2.0;
                c.input_gate_bias = -0.5;
                c.exp_gate_scale = 6.0;
                c.exp_clamp = 8.0;
                c
            },
        },
        SweepCase {
            label: "COMBO: paper-like (ws=0.02, fb=1.0, egs=2.0, ec=20.0, eps=1e-6)".into(),
            cfg: base().with_stability(0.02, 1.0, 1e-6),
        },
    ]
}

// ─── Main ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let device = Device::Cpu;
    let args: Vec<String> = env::args().collect();
    let max_cases: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 = todos
    let train_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let dim = 128;
    let heads = 4;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        mLSTM PARAMETER SWEEP — STRESS TEST                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("  dim={}, heads={}, train_steps={}", dim, heads, train_steps);
    println!();

    let cases = build_sweep_cases(dim, heads);
    let total = if max_cases > 0 && max_cases < cases.len() { max_cases } else { cases.len() };

    // Tabla de resultados para el resumen final
    let mut results: Vec<(String, bool, f32, f32, f32, f32)> = Vec::new();

    for (i, case) in cases.iter().take(total).enumerate() {
        println!("━━━ Caso {}/{}: {} ━━━", i + 1, total, case.label);
        println!("   Params: ws={:.4}, fb={:.2}, igb={:.2}, eps={:.1e}, ec={:.1}, egs={:.1}, nc=[{:.1e},{:.1e}], sep_bias={}",
            case.cfg.weight_stdev,
            case.cfg.forget_bias,
            case.cfg.input_gate_bias,
            case.cfg.epsilon,
            case.cfg.exp_clamp,
            case.cfg.exp_gate_scale,
            case.cfg.norm_clamp_min,
            case.cfg.norm_clamp_max,
            case.cfg.use_separate_bias,
        );

        // 1. Test de estabilidad
        let stable = match run_stability(&case.cfg, &device) {
            Ok(s) => s,
            Err(e) => {
                println!("   ❌ Error en estabilidad: {:?}", e);
                false
            }
        };

        if !stable {
            println!("   ❌ INESTABLE — salteando entrenamiento");
            results.push((case.label.clone(), false, f32::NAN, f32::NAN, f32::NAN, f32::NAN));
            println!();
            continue;
        }
        println!("   ✅ Estabilidad OK");

        // 2. Entrenamiento
        match train_once(&case.cfg, train_steps, &device) {
            Ok(((l0o, lfo), (l0p, lfp))) => {
                let delta_orig = if l0o > 0.0 { (lfo - l0o) / l0o * 100.0 } else { 0.0 };
                let delta_par  = if l0p > 0.0 { (lfp - l0p) / l0p * 100.0 } else { 0.0 };

                let status_o = if lfo < l0o { "✅ MEJORÓ" } else if lfo.is_nan() { "❌ NaN" } else { "⚠️ NO mejoró" };
                let status_p = if lfp < l0p { "✅ MEJORÓ" } else if lfp.is_nan() { "❌ NaN" } else { "⚠️ NO mejoró" };

                println!("   [ORIGINAL] Loss: {:.6} → {:.6} ({:+.1}%) {}", l0o, lfo, delta_orig, status_o);
                println!("   [PARIDAD ] Loss: {:.6} → {:.6} ({:+.1}%) {}", l0p, lfp, delta_par, status_p);

                results.push((case.label.clone(), true, l0o, lfo, l0p, lfp));
            },
            Err(e) => {
                println!("   ❌ Error en entrenamiento: {:?}", e);
                results.push((case.label.clone(), true, f32::NAN, f32::NAN, f32::NAN, f32::NAN));
            }
        }
        println!();
    }

    // ─── RESUMEN FINAL ───────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESUMEN DE RESULTADOS                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:>3} │ {:<50} │ {:>5} │ {:>10} │ {:>10} ║", "#", "Configuración", "Estab", "Δ% Orig", "Δ% Par");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");

    // Ordenar por mejor convergencia en tarea original
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| {
        let da = if a.2 > 0.0 && !a.3.is_nan() { (a.3 - a.2) / a.2 } else { f32::MAX };
        let db = if b.2 > 0.0 && !b.3.is_nan() { (b.3 - b.2) / b.2 } else { f32::MAX };
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, (label, stable, l0o, lfo, l0p, lfp)) in sorted.iter().enumerate() {
        let stab_str = if *stable { " ✅ " } else { " ❌ " };
        let delta_o = if *l0o > 0.0 && !lfo.is_nan() { format!("{:>+9.1}%", (lfo - l0o) / l0o * 100.0) } else { "   N/A   ".into() };
        let delta_p = if *l0p > 0.0 && !lfp.is_nan() { format!("{:>+9.1}%", (lfp - l0p) / l0p * 100.0) } else { "   N/A   ".into() };
        let trunc_label: String = label.chars().take(50).collect();
        println!("║ {:>3} │ {:<50} │{} │ {} │ {} ║", rank + 1, trunc_label, stab_str, delta_o, delta_p);
    }

    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("🏆 El mejor caso es el que tiene el Δ% más negativo (mayor reducción de loss).");
    println!("   Casos con NaN/Inf son numéricamente inestables con esos parámetros.");

    Ok(())
}
