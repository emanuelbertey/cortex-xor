[🇪🇸 Versión en Español](#cortex-xor-sala-de-pruebas-experimental-de-rnns)

# Cortex-XOR: Experimental RNN Playground

This repository serves as an experimental testing ground for advanced Recurrent Neural Network (RNN) architectures implemented in Rust using the Candle framework. It is currently a work in progress and does not represent a final, concrete product.

## Core Components Tested
We are actively implementing and testing the following blocks, often in stacked configurations:

*   **sLSTM (Scalar LSTM):** Features exponential gating to improve scalar memory capabilities.
*   **mLSTM (Matrix LSTM):** Utilizes a matrix memory state to significantly enhance storage capacity.
*   **minGRU:** A minimal, efficient Gated Recurrent Unit.
*   **minLSTM:** A simplified and optimized variant of the traditional LSTM.

## Key Features & Goals
*   **Stacked Architectures:** We are experimenting with deep networks by stacking these blocks in various patterns (e.g., alternating sLSTM and mLSTM layers).
*   **Parallel Execution:** We are developing parallelized versions of these blocks to maximize performance and training efficiency.
*   **Paper Alignment:** The implementation aims to follow the original "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024) and related research papers very closely, though exact adherence is sometimes maintained flexibly for experimental purposes and practical optimization.

## Training & Methodology
*   **Datasets:** The models are primarily trained and benchmarked on **Shakespeare** and **TinyStories V2** datasets.
*   **Process:** The project is in a state of constant flux, with ongoing stability tests, parameter sweeps, and continuous code improvements.

## Performance Demo: xLSTM vs minLSTM
A comparison demonstrating the trade-off between "Brain" (accuracy) and "Time" (efficiency):

*   **xLSTM** (More "Brain"):
    *   **Accuracy:** ~22
    *   **Time (100 tokens):** 0.70s
    *   **Note:** Higher resource consumption.
    *   **Sample:**
        > king,
        > And told your honour, my lord, I know not
        > com, and you to me by my gracious lord.
        > DUKE VINCENTIO:
        > Ah, and the wounds for this modnerousous and strong to my heart,

*   **minLSTM** (More Speed):
    *   **Accuracy:** ~19
    *   **Time (100 tokens):** 0.20s
    *   **Efficiency:** Consumes ~50% less RAM than xLSTM.
    *   **Sample:**
        > king, as the sigh bedildier
        > Stucucess, towningman:
        > Wealmpove
        > For:
        > Thybed to the master's death.


---

# Cortex-XOR: Sala de Pruebas Experimental de RNNs

Este repositorio sirve como una sala de pruebas experimental para arquitecturas avanzadas de Redes Neuronales Recurrentes (RNN) implementadas en Rust utilizando el framework Candle. Actualmente es un trabajo en progreso y no representa un producto final o concreto.

## Componentes Principales Probados
Estamos implementando y probando activamente los siguientes bloques, a menudo en configuraciones apiladas:

*   **sLSTM (Scalar LSTM):** Incorpora "gating" (compuertas) exponencial para mejorar la capacidad de memoria escalar.
*   **mLSTM (Matrix LSTM):** Utiliza un estado de memoria matricial para aumentar significativamente la capacidad de almacenamiento.
*   **minGRU:** Una Unidad Recurrente con Compuertas (GRU) minimalista y eficiente.
*   **minLSTM:** Una variante simplificada y optimizada del LSTM tradicional.

## Características Clave y Objetivos
*   **Arquitecturas Apiladas:** Estamos experimentando con redes profundas apilando estos bloques en varios patrones (por ejemplo, alternando capas de sLSTM y mLSTM).
*   **Ejecución Paralela:** Estamos desarrollando versiones paralizadas de estos bloques para maximizar el rendimiento y la eficiencia del entrenamiento.
*   **Alineación con el Paper:** La implementación intenta seguir muy de cerca el paper original "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024) y las investigaciones relacionadas, aunque no es exacto al 100%, permitiendo flexibilidad para la experimentación y optimización práctica.

## Entrenamiento y Metodología
*   **Datasets:** Los modelos se entrenan y evalúan principalmente con los datasets de **Shakespeare** y **TinyStories V2**.
*   **Proceso:** El proyecto está en un estado de cambio constante, con pruebas de estabilidad en curso, barridos de parámetros y mejoras continuas en el código.

## Demostración de Rendimiento: xLSTM vs minLSTM
Una comparación que demuestra el compromiso entre "Cerebro" (precisión) y "Tiempo" (eficiencia):

*   **xLSTM** (Más "Cerebro"):
    *   **Precisión:** ~22
    *   **Tiempo (100 tokens):** 0.70s
    *   **Nota:** Mayor consumo de recursos.
    *   **Muestra:**
        > king,
        > And told your honour, my lord, I know not
        > com, and you to me by my gracious lord.
        > DUKE VINCENTIO:
        > Ah, and the wounds for this modnerousous and strong to my heart,

*   **minLSTM** (Más Velocidad):
    *   **Precisión:** ~19
    *   **Tiempo (100 tokens):** 0.20s
    *   **Eficiencia:** Consume ~50% menos RAM que xLSTM.
    *   **Muestra:**
        > king, as the sigh bedildier
        > Stucucess, towningman:
        > Wealmpove
        > For:
        > Thybed to the master's death.
