#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization

This example demonstrates how to use xLSTM for text generation
using a simple character-level tokenizer that can be saved/loaded as JSON.

*/

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use anyhow::Result;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::collections::HashSet;
use std::time::Instant;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

use xlstm::{LstmType, XLstm, XLstmconfig, BlockType};
use rand::Rng;

/// Tokenizador profesional usando la librería 'tokenizers' de Hugging Face
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self> {
        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut tokenizer = HFTokenizer::new(model);

        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '▁',
            PrependScheme::Always,
            true,
        )));

        let mut alphabet = HashSet::new();
        alphabet.insert('\n');
        alphabet.insert(' ');

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(0)
            .initial_alphabet(alphabet)
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);

        let temp_file = "temp_train.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| anyhow::anyhow!(e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    /// Guarda el tokenizador en un archivo
    pub fn save(&self, path: &str) -> Result<()> {
        self.tokenizer.save(path, true)
            .map_err(|e| anyhow::anyhow!("Error al guardar: {}", e))?;
        println!("Tokenizador guardado en: {}", path);
        Ok(())
    }

    /// Carga el tokenizador desde un archivo
    pub fn load(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Error al cargar: {}", e))?;
        println!("Tokenizador cargado desde: {}", path);
        Ok(Self { tokenizer })
    }

    /// Convierte texto a índices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    /// Convierte índices a texto
    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }

    /// Obtiene el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Obtiene el string de un token por su índice
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

/// Crea un batch de entrenamiento (indices) para usar con Embedding
fn create_batch(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize, // Añadido stride explícitamente si se usa dentro
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + (i * stride);
        for j in 0..seq_length {
            let idx = current_start + j;
            x_indices.push(tokens[idx] as u32);
            y_indices.push(tokens[idx + 1] as u32);
        }
    }

    let x = Tensor::from_vec(x_indices, (batch_size, seq_length), device)?;
    let y = Tensor::from_vec(y_indices, (batch_size, seq_length), device)?;

    Ok((x, y))
}

/*
/// Selecciona un token usando muestreo estocástico con Top-K y temperatura
fn sample_from_logits(logits: &Tensor, temperature: f32) -> Result<usize> {
    // logits: [1, vocab_size]
    let logits = logits.squeeze(0)?;
    let vocab_size = logits.dim(0)?;
    
    // To Vec for manual sampling (simpler than tensor operations for top-k sampling for now)
    let logits_vec = logits.to_vec1::<f32>()?;
    
    let mut probs_vec: Vec<(usize, f32)> = logits_vec.iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    // Softmax is applied later in sampling logic or here?
    // The original code applied softmax first, then top-k on probs.
    // Let's replicate logic: Softmax then Top-K.
    
    // We can do softmax on tensor first
    let probs = softmax(&logits, 0)?;
    let probs_vec_tensor = probs.to_vec1::<f32>()?;
    let mut probs_indexed: Vec<(usize, f32)> = probs_vec_tensor.into_iter().enumerate().collect();


    // --- TOP-K ---
    // Ordenar de mayor a menor probabilidad
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Solo nos quedamos con los 5 o 10 mejores candidatos
    let k = 5; 
    let top_k_probs = &probs_indexed[..k.min(vocab_size)];
    
    // Extraer solo los pesos para el muestreo
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    // --------------------

    // Si la temperatura es muy baja, actuar de forma determinista (Greedy)
    if temperature <= 1e-6 {
        return Ok(indices[0]);
    }

    // Aplicar temperatura sobre el Top-K
    for p in weights.iter_mut() {
        // p is probability. log(p)/temp -> exp
         *p = (p.max(1e-10).ln() / temperature).exp();
    }

    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng(); 
    let sample: f32 = rng.random::<f32>() * sum;
    let mut cumulative = 0.0;

    for (i, &p) in weights.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative {
            return Ok(indices[i]);
        }
    }

    Ok(indices[0])
}
    */
fn sample_from_logits(logits: &Tensor, temperature: f32, top_k: usize, top_p: f32, greedy_threshold: f32) -> Result<usize> {
    let logits = logits.squeeze(0)?;
    let vocab_size = logits.dim(0)?;
    let scaled_logits = (&logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let mut probs_indexed: Vec<(usize, f32)> = probs.to_vec1::<f32>()?.into_iter().enumerate().collect();
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_p = probs_indexed.first().map(|(_, p)| *p).unwrap_or(0.0);
    if max_p >= greedy_threshold {
        return Ok(probs_indexed[0].0);
    }
    let k = top_k.min(vocab_size);
    let mut cumulative = 0.0f32;
    let mut filtered: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (idx, p) in probs_indexed.into_iter().take(k) {
        if cumulative + p <= top_p || filtered.is_empty() {
            filtered.push((idx, p));
            cumulative += p;
        }
    }
    let indices: Vec<usize> = filtered.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = filtered.iter().map(|(_, p)| *p).collect();
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng();
    let mut sample: f32 = rng.random::<f32>() * sum;
    for (i, &p) in weights.iter().enumerate() {
        if sample <= p {
            return Ok(indices[i]);
        }
        sample -= p;
    }
    Ok(indices[0])
}

fn sample_with_constraints(logits: &Tensor, tokenizer: &Tokenizer, temperature: f32, top_k: usize, top_p: f32, greedy_threshold: f32) -> Result<usize> {
    let logits = logits.squeeze(0)?;
    let vocab_size = logits.dim(0)?;
    let scaled_logits = (&logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let mut probs_indexed: Vec<(usize, f32)> = probs.to_vec1::<f32>()?.into_iter().enumerate().collect();
    probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_p = probs_indexed.first().map(|(_, p)| *p).unwrap_or(0.0);
    if max_p >= greedy_threshold {
        return Ok(probs_indexed[0].0);
    }
    let k = top_k.min(vocab_size);
    let mut cumulative = 0.0f32;
    let mut filtered: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (idx, p) in probs_indexed.into_iter().take(k) {
        if cumulative + p <= top_p || filtered.is_empty() {
            filtered.push((idx, p));
            cumulative += p;
        }
    }
    // Primera pasada: evitar letras sueltas y tokens que sean solo marcadores
    for (idx, _) in &filtered {
        if let Some(tok) = tokenizer.id_to_token(*idx) {
            let t = tok.trim_start_matches('▁').trim_start_matches('Ġ').replace("Ċ", "");
            let is_single_alpha = t.chars().count() == 1 && t.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false);
            let is_marker_only = tok.starts_with('▁') || tok.starts_with('Ġ') || tok.contains('Ċ');
            if !is_single_alpha && !is_marker_only {
                return Ok(*idx);
            }
        }
    }
    // Segunda pasada: permitir espacio ('▁'/'Ġ') si no hubo mejor opción
    for (idx, _) in &filtered {
        if let Some(tok) = tokenizer.id_to_token(*idx) {
            if tok.starts_with('▁') || tok.starts_with('Ġ') {
                return Ok(*idx);
            }
        }
    }
    // Fallback probabilístico
    let indices: Vec<usize> = filtered.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = filtered.iter().map(|(_, p)| *p).collect();
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng();
    let mut sample: f32 = rng.random::<f32>() * sum;
    for (i, &p) in weights.iter().enumerate() {
        if sample <= p {
            return Ok(indices[i]);
        }
        sample -= p;
    }
    Ok(indices[0])
}


/// Genera texto de forma recurrente manteniendo el estado interno del modelo
fn generate_text(
    model: &XLstm,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &Device,
) -> Result<String> {
    let mut generated_ids: Vec<usize> = tokenizer.encode(seed_text);
    let seed_tokens = tokenizer.encode(seed_text);
    
    if seed_tokens.is_empty() {
        return Ok(seed_text.to_string());
    }

    // 1) Procesar la semilla completa para inicializar estado
    let seed_indices: Vec<u32> = seed_tokens.iter().map(|&t| t as u32).collect();
    let input_seed = Tensor::from_vec(seed_indices, (1, seed_tokens.len()), device)?;
    let (seed_output, mut state) = model.forward(&input_seed, None)?;
    let first_logits = seed_output.narrow(1, seed_tokens.len() - 1, 1)?.squeeze(1)?.detach();
    let mut last_token = sample_with_constraints(&first_logits, tokenizer, 0.7, 50, 0.9, 0.95)?;

    // 2) Loop autoregresivo: pasar solo el nuevo token y usar el estado acumulado
    for _ in 0..length {
        let input = Tensor::from_vec(vec![last_token as u32], (1, 1), device)?;
        let (step_output, next_state) = model.forward(&input, Some(state))?;
        state = next_state.into_iter().map(|s| s.map(|st| st.detach())).collect();

        let step_logits = step_output.squeeze(1)?.detach();
        last_token = sample_with_constraints(&step_logits, tokenizer, 0.7, 50, 0.9, 0.95)?;
        generated_ids.push(last_token);
    }

    let mut out = String::new();
    for &id in &generated_ids {
        if let Some(mut t) = tokenizer.id_to_token(id) {
            if t.contains('Ċ') { out.push(' '); continue; }
            if t.starts_with('▁') || t.starts_with('Ġ') {
                out.push(' ');
                t = t.trim_start_matches('▁').trim_start_matches('Ġ').to_string();
            }
            t = t.replace("\r\n", "").replace('\r', "");
            out.push_str(&t);
        }
    }
    out = out.replace('\n', " ");
    out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(out)
}

fn main() -> Result<()> {
    println!("xLSTM (mLSTM) Text Generation con Tokenizador (Candle)");
    println!("====================================================\n");

    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin mlstmchat -- <archivo.txt>");
        eprintln!("Ejemplo: cargo run --bin mlstmchat -- input.txt");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer_mlstm.json";
    let model_path = "xlstm_chat_model_mlstm.safetensors";

    let target_vocab_size = 1024;

    let tokenizer = if Path::new(tokenizer_path).exists() {
        println!("Cargando tokenizador existente...");
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Entrenando nuevo tokenizador profesional (BPE) desde {}...", text_file);
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

println!("\n--- VERIFICACIÓN DE IDENTIDAD DE TOKENS ---");
let texto_verificacion = " \n"; // Un espacio y un salto de línea
let ids = tokenizer.encode(texto_verificacion);

for id in ids {
    let contenido = tokenizer.decode(&[id]);
    // Esto te dirá exactamente qué ID tiene el espacio y cuál el salto
    if contenido.contains(' ') {
        println!("ID: {:<5} | Representa: [ESPACIO SEGURO]", id);
    } else if contenido.contains('\n') {
        println!("ID: {:<5} | Representa: [SALTO DE LINEA SEGURO]", id);
    } else {
        println!("ID: {:<5} | Representa: '{}'", id, contenido);
    }
}
println!("-------------------------------------------\n");

let prueba = tokenizer.encode(" ");
println!("DEBUG ESPACIO: {:?}", prueba);

let prueba_salto = tokenizer.encode("\n");
println!("DEBUG SALTO: {:?}", prueba_salto);
    println!("Tamaño del vocabulario: {}\n", tokenizer.vocab_size());

    println!("Cargando texto de entrenamiento...");
    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    println!("Tokens totales: {}\n", tokens.len());

    let vocab_size = tokenizer.vocab_size();
    let hidden_size = 512; 
    let num_layers = 1;
    let num_blocks = 2;
    let output_size = vocab_size; 
    let  mut dropout = 0.0;

    let seq_length = 128; 
    let batch_size = 16; 
    let stride = 128;     
    let num_epochs = 50;
    let num_heads = 2;

    println!("Configuración del modelo:");
    println!("  Bloques: {}", num_blocks);
    println!("  Hidden size: {}", hidden_size);
    println!("  Seq length: {}", seq_length);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}\n", num_epochs);

    let device = Device::Cpu;

     let config = XLstmconfig::new(hidden_size, hidden_size, num_layers, num_blocks, output_size)
        .with_vocab_size(vocab_size)
        .with_dropout(dropout)
        .with_num_heads(num_heads)
        .with_lstm_type(LstmType::MLSTM)
        .with_use_projection(true);   

    let model_file_path = Path::new(model_path);
    let existe_modelo = model_file_path.exists();
    
    let mut continuar_entrenamiento = false;
    if existe_modelo {
        print!("¿Deseas seguir entrenando el modelo cargado? (s/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "s" {
            continuar_entrenamiento = true;
         }/* else {
            continuar_entrenamiento = false;
         }*/
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    //let model = config.init(vb)?;
    let mut model = config.init(vb)?;

    if existe_modelo {
         if !continuar_entrenamiento {
             println!("¡Modelo encontrado! Cargando pesos para generación...");
             varmap.load(model_path)?;
             println!("Modelo cargado exitosamente!\n");
         } else {
             println!("Cargando modelo previo para continuar entrenamiento...");
             varmap.load(model_path)?;
         }
    } else {
         println!("No se encontró modelo guardado. Iniciando entrenamiento desde cero...\n");
    }
    
    // Logic from main.rs training loop, adapted
    if !existe_modelo || continuar_entrenamiento {

        if !tokens.is_empty() {
             let first_token_idx = tokens[0];
             let first_token_str = tokenizer.id_to_token(first_token_idx).unwrap_or("?".to_string());
             println!("--- INSPECCIÓN DE EMBEDDING ---");
             println!("  Token Index: {}", first_token_idx);
             println!("  Token Str: '{}'", first_token_str);
             println!("-----------------------------\n");
        }

        let num_sequences = tokens.len().saturating_sub(seq_length + 1);
        let num_actual_sequences = num_sequences / stride;

        // Group parameters for optimizers (as in main.rs)
        let parsed_block_types = match config.lstm_type {
            LstmType::SLSTM => vec![BlockType::SLSTM; num_blocks],
            LstmType::MLSTM => vec![BlockType::MLSTM; num_blocks],
            LstmType::Alternate => (0..num_blocks)
                .map(|i| if i % 2 == 0 { BlockType::SLSTM } else { BlockType::MLSTM })
                .collect(),
            LstmType::Custom(ref types) => types.clone(),
        };

        let mut slstm_params = Vec::new();
        let mut mlstm_params = Vec::new();
        let mut other_params = Vec::new();

        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if name.starts_with("block_") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Some(block_part) = parts.first() {
                    if let Some(idx_str) = block_part.strip_prefix("block_") {
                         if let Ok(idx) = idx_str.parse::<usize>() {
                             if idx < parsed_block_types.len() {
                                 match parsed_block_types[idx] {
                                     BlockType::SLSTM => slstm_params.push(var.clone()),
                                     BlockType::MLSTM => mlstm_params.push(var.clone()),
                                 }
                             } else { other_params.push(var.clone()); }
                         } else { other_params.push(var.clone()); }
                    } else { other_params.push(var.clone()); }
                } else { other_params.push(var.clone()); }
            } else { other_params.push(var.clone()); }
        }
        drop(data); // release lock before training

        // Tasas de aprendizaje recomendadas para xLSTM: 
        // sLSTM suele tolerar LRs más altas, mLSTM requiere más cuidado.
        let mut optim_slstm = AdamW::new(slstm_params, ParamsAdamW { lr: 2e-4, ..Default::default() })?;
       // let mut optim_mlstm = AdamW::new(mlstm_params, ParamsAdamW { lr: 8e-4, ..Default::default() })?;
        let mut optim_other = AdamW::new(other_params, ParamsAdamW { lr: 2e-4, ..Default::default() })?;


        
model.print_architecture();
        let lr_max = 1e-4;
        let lr_min = 8e-5;
        let mut aumentando = false; // Control de dirección
        let step_factor = 1.0;      // Qué tan rápido cambia
        let mut current_lr = 4e-5;
        let mut optim_mlstm = AdamW::new(mlstm_params.clone(), ParamsAdamW { 
            lr: current_lr, 
            ..Default::default() 
        })?;


        println!("Iniciando entrenamiento...\n");

        let num_batches = num_actual_sequences.div_ceil(batch_size);
        dropout = 0.0;
        for epoch in 0..num_epochs {
            let mut total_loss = 0.0f32;
            let mut num_losses = 0;
            let mut correct = 0;
            let mut total = 0;
           let mut current_state = None;
            for batch_idx in 0..num_batches {
                let epoch_start = Instant::now();
                let current_batch_start_seq = batch_idx * batch_size;
                let current_batch_size = (batch_size).min(num_actual_sequences - current_batch_start_seq);

                    if current_batch_size == 0 { break; }

                let (input_batch, target_batch) = create_batch(
                    &tokens,
                    current_batch_start_seq * stride,
                    current_batch_size,
                    seq_length,
                    stride,
                    &device,
                )?;
                if epoch == 0 && batch_idx == 0 {
                    let input_ids = input_batch.narrow(0, 0, 1)?.squeeze(0)?.to_vec1::<u32>()?;
                    let target_ids = target_batch.narrow(0, 0, 1)?.squeeze(0)?.to_vec1::<u32>()?;
                    let input_usize: Vec<usize> = input_ids.iter().map(|&x| x as usize).collect();
                    let target_usize: Vec<usize> = target_ids.iter().map(|&x| x as usize).collect();
                    // Reconstrucción desde tokens (sin decode HF), respetando límites de palabra
                    let input_tokens: Vec<String> = input_usize
                        .iter()
                        .map(|&id| tokenizer.id_to_token(id).unwrap_or("?".to_string()))
                        .collect();
                    let target_tokens: Vec<String> = target_usize
                        .iter()
                        .map(|&id| tokenizer.id_to_token(id).unwrap_or("?".to_string()))
                        .collect();
                    let mut input_text = String::new();
                    for mut t in input_tokens {
                        if t.contains('Ċ') { input_text.push('\n'); continue; }
                        let has_space = t.starts_with('▁') || t.starts_with('Ġ');
                        if has_space {
                            input_text.push(' ');
                            t = t.trim_start_matches('▁').trim_start_matches('Ġ').to_string();
                        }
                        t = t.replace("\r\n", "").replace('\r', "");
                        input_text.push_str(&t);
                    }
                    let mut target_text = String::new();
                    for mut t in target_tokens {
                        if t.contains('Ċ') { target_text.push('\n'); continue; }
                        let has_space = t.starts_with('▁') || t.starts_with('Ġ');
                        if has_space {
                            target_text.push(' ');
                            t = t.trim_start_matches('▁').trim_start_matches('Ġ').to_string();
                        }
                        t = t.replace("\r\n", "").replace('\r', "");
                        target_text.push_str(&t);
                    }
                    input_text = input_text.replace('\n', " ");
                    target_text = target_text.replace('\n', " ");
                    input_text = input_text.split_whitespace().collect::<Vec<_>>().join(" ");
                    target_text = target_text.split_whitespace().collect::<Vec<_>>().join(" ");
                    println!("\n[Epoch 1] Batch de entrenamiento (texto limpio):");
                    println!("  Input Texto:  {}", input_text);
                    println!("  Target Texto: {}", target_text);
                }
/*
                if batch_idx == 0 {
                    let (_, warm_state) = model.forward(&input_batch, None)?;
                    current_state = Some(warm_state.into_iter().map(|s| s.map(|state| state.detach())).collect());
                }*/
                {
                    let (logits, next_state) = model.forward(&input_batch, current_state)?;
                    let logits_flat = logits.reshape((current_batch_size * seq_length, vocab_size))?;
                    let target_flat = target_batch.reshape((current_batch_size * seq_length,))?;
                    let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
                    total_loss += loss.to_scalar::<f32>()?;
                    num_losses += 1;
                    let last_logits = logits.narrow(1, seq_length - 1, 1)?.squeeze(1)?;
                    let last_targets = target_batch.narrow(1, seq_length - 1, 1)?.squeeze(1)?;
                    let last_preds = last_logits.argmax(1)?;
                    let correct_count = last_preds.eq(&last_targets)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
                    correct += correct_count;
                    total += current_batch_size;
                    let grads = loss.backward()?;
                    optim_slstm.step(&grads)?;
                    optim_mlstm.step(&grads)?;
                    optim_other.step(&grads)?;
                    current_state = Some(next_state.into_iter().map(|s| s.map(|t| t.detach())).collect());
                }

                if batch_idx % 1 == 0 || batch_idx == num_batches - 1 {
                    let elapsed = epoch_start.elapsed().as_secs_f32();
                    print!("\r  -> Batch [{}/{}] Loss: {:.4} Acc: {:.2}% ({:.1}s)", 
                        batch_idx + 1, num_batches, total_loss / (num_losses as f32),
                        100.0 * correct as f32 / total as f32, elapsed);
                    io::stdout().flush().unwrap();
                
                }
            
                 if batch_idx % 50 == 0 && batch_idx > 0 {
                    let factor = step_factor as f32; //  errores

                    if aumentando {

                        dropout /= factor;           // 1. Actualizamos la variable local
                        model.dropout = dropout;     // 2. Se la pasamos al model
                        current_lr /= step_factor; // El LR suele ser f64, está bien
                        if current_lr >= lr_max {
                            current_lr = lr_max;
                            aumentando = false; 
                        }
                    } else {
                            dropout *= factor;           // 1. Actualizamos la variable local 
                        current_lr *= step_factor; 
                        if current_lr <= lr_min {
                            current_lr = lr_min;
                            aumentando = true; 
                        }
                    }
                    dropout = 0.0;//dropout.clamp(-0.0, 0.0);
                    model.blocks.iter_mut().for_each(|b| b.dropout_prob = dropout);
            
                    println!(
                        "\n[CYCLIC SCHEDULER] LR: {:.2e} | Dropout: {:.4} | Dirección: {}", 
                        current_lr, 
                       dropout, //
                        if aumentando { "Sube ↑" } else { "Baja ↓" }
                    );
                } 

                optim_mlstm = AdamW::new(mlstm_params.clone(),ParamsAdamW {lr: current_lr,..Default::default() }  )?;
                                        
            }
            println!();

            let avg_loss = total_loss / num_losses as f32;
            let accuracy = 100.0 * correct as f32 / total as f32;

            println!("Epoch [{:3}/{}], Loss: {:.4}, Accuracy: {:.2}%", epoch + 1, num_epochs, avg_loss, accuracy);

            // Save per epoch
            varmap.save(model_path)?;

            // Generate sample
             if epoch % 1 == 0 {
                let mut rng = rand::rng();
                let mut start = if tokens.len() > 40 {
                    rng.random_range(0..tokens.len() - 40)
                } else { 0 };
                for _ in 0..20 {
                    if start >= tokens.len() { break; }
                    let tok = tokenizer.id_to_token(tokens[start]).unwrap_or_default();
                    if tok.starts_with('▁') || tok.starts_with('Ġ') || tok.contains('Ċ') { break; }
                    start += 1;
                }
                let take = 30.min(tokens.len().saturating_sub(start));
                let seed_slice: &[usize] = &tokens[start..start + take];
                let mut seed_clean = tokenizer.decode(seed_slice);
                seed_clean = seed_clean.replace('▁', " ").replace('Ġ', " ").replace("Ċ", "\n").replace("\r\n", "\n");
                seed_clean = seed_clean.split_whitespace().collect::<Vec<_>>().join(" ");
                println!("  -> Generando con semilla al azar: '{}'", seed_clean);
                let generated = generate_text(&model, &tokenizer, &seed_clean, 100, &device)?;
                let generated_clean = generated.replace('▁', " ").replace('Ġ', " ").replace("\r\n", "\n");
                println!("  Generado: {}\n", generated_clean);
            }
        }
        println!("\n¡Entrenamiento completado!");
    } else {
        // Just inference mode
    }

    // Modo interactivo - Loop para generar texto
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO - GENERACIÓN DE TEXTO         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("Comandos:");
    println!("  - Escribe un texto semilla y presiona Enter para generar");
    println!("  - Escribe 'salir' o 'exit' para terminar");
    println!("  - Escribe 'auto' para generar con semilla automática");
    println!("  - Escribe 'len N' para cambiar longitud de generación (tokens)\n");

    let mut gen_length: usize = 200;

    loop {
        print!("Semilla > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") { break; }

        if input.to_lowercase().starts_with("len") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() == 1 {
                println!("Longitud actual de generación: {} tokens\n", gen_length);
            } else if parts.len() >= 2 {
                match parts[1].parse::<usize>() {
                    Ok(n) if n > 0 && n <= 20000 => {
                        gen_length = n;
                        println!("Nueva longitud de generación establecida en {} tokens\n", gen_length);
                    }
                    Ok(_) => {
                        println!("Por favor usa un valor entre 1 y 20000.\n");
                    }
                    Err(_) => {
                        println!("Formato inválido. Usa: len 200\n");
                    }
                }
            }
            continue;
        }

         let seed = if input.eq_ignore_ascii_case("auto") {
             // We need 'text' but it might not be loaded if we didn't train.
             // If not trained, we might fail here. But simpler to assume we have text or don't support auto.
             if tokens.len() > 20 {
                 let seed_tokens: Vec<usize> = tokens[0..20].to_vec();
                 tokenizer.decode(&seed_tokens)
             } else {
                 "Once upon a time".to_string()
             }
        } else {
            input.to_string()
        };

        println!("\nGenerando...");
        let generated = generate_text(&model, &tokenizer, &seed, gen_length, &device)?;
        println!("Generado: {}\n", generated);
    }

    Ok(())
}
