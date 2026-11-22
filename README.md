# Proyecto 2: Fine-tuning de modelos Transformer para tareas de NLP

Este proyecto aplica *fine-tuning* supervisado a modelos Transformer pre-entrenados para resolver tres problemas de
Procesamiento de Lenguaje Natural, uno por cada familia de tareas: clasificación de secuencias, etiquetado de
secuencias y generación *sequence-to-sequence*. En cada caso se describe y explora la base de datos, se buscan los
hiperparámetros sobre una muestra, se entrena el modelo con la data completa y se compara contra una línea base
clásica.

El desarrollo completo está en el notebook [`nlp_finetuning_three_tasks.ipynb`](./nlp_finetuning_three_tasks.ipynb)
([abrir en Google Colab](https://colab.research.google.com/github/AndiEspejo/emotions-skills-detox-nlp/blob/main/nlp_finetuning_three_tasks.ipynb)).

## Problemas y resultados

| Tarea | Base de datos | Modelo | Resultado en test | Línea base clásica |
| :--- | :--- | :--- | :--- | :--- |
| Detección de emociones en comentarios (multi-etiqueta, 28 clases) | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | RoBERTa-base | Macro-F1 **0.523** · Micro-F1 **0.603** | TF-IDF + regresión logística: 0.420 · 0.554 |
| Extracción de habilidades y conocimientos en ofertas de empleo | [SkillSpan](https://huggingface.co/datasets/jjzha/skillspan) | DeBERTa-v3-base | F1 por entidad **0.628** | Diccionario de entidades de entrenamiento: 0.184 |
| Reescritura de comentarios tóxicos en lenguaje respetuoso | [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) | BART-base | BLEU **66.4** · 86.9 % no tóxico | Eliminación de palabras ofensivas: 63.4 · 85.8 % |

Los modelos finales están disponibles en Hugging Face:
[`roberta-base-goemotions-udea`](https://huggingface.co/jeffespejo/roberta-base-goemotions-udea),
[`deberta-v3-base-skillspan-udea`](https://huggingface.co/jeffespejo/deberta-v3-base-skillspan-udea) y
[`bart-base-paradetox-udea`](https://huggingface.co/jeffespejo/bart-base-paradetox-udea).

## Conclusiones

- El *learning rate* fue el hiperparámetro más determinante en las tres tareas. Con valores bajos y pocas
  actualizaciones por época, los modelos prácticamente no aprendieron.
- En la extracción de habilidades se compararon BERT, JobBERT (pre-entrenado en ofertas de empleo) y DeBERTa-v3.
  DeBERTa-v3 obtuvo el mejor resultado, lo que indica que la arquitectura aportó más que el pre-entrenamiento en el dominio.
- Los modelos generalizan más allá de los ejemplos vistos: el 63 % de las entidades del conjunto de prueba no aparece en
  entrenamiento, y el modelo recupera más de la mitad de ellas, mientras que el diccionario no recupera ninguna.
- En la desintoxicación, eliminar palabras ofensivas resultó una línea base competitiva. La ventaja del modelo se
  concentra en los casos que requieren reformular la frase.

## Ejecución

El notebook está pensado para Google Colab con GPU T4: la primera celda instala las dependencias que falten. Para
ejecutarlo localmente se requiere Python 3.10+ y los paquetes de [`requirements.txt`](./requirements.txt). Con la
variable de entorno `SMOKE=1` se ejecuta una versión reducida en pocos minutos, útil para verificar el entorno.

---

Proyecto 2 del curso *Fundamentos de Procesamiento de Lenguaje Natural*, Ingeniería de Sistemas, Universidad de
Antioquia (Prof. Antonio Jesús Tamayo Herrera). Autor: Jefferson Espejo.
