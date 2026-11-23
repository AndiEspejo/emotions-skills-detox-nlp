# Proyecto 2: Fine-tuning de modelos Transformer para tareas de NLP

Este proyecto aplica *fine-tuning* supervisado a modelos Transformer pre-entrenados para resolver tres problemas de
Procesamiento de Lenguaje Natural, uno por cada familia de tareas: clasificación de secuencias, etiquetado de
secuencias y generación *sequence-to-sequence*. Las tres bases de datos se eligieron por tener una aplicación concreta:
reconocer emociones en comentarios, extraer requisitos de ofertas de empleo y reescribir mensajes ofensivos.

El desarrollo completo está en el notebook [`nlp_finetuning_three_tasks.ipynb`](./nlp_finetuning_three_tasks.ipynb)
([abrir en Google Colab](https://colab.research.google.com/github/AndiEspejo/emotions-skills-detox-nlp/blob/main/nlp_finetuning_three_tasks.ipynb)).

## Problemas abordados

| Tarea | Tipo de problema | Base de datos | Entrada → salida | Modelo |
| :--- | :--- | :--- | :--- | :--- |
| Detección de emociones | Clasificación de secuencias, multi-etiqueta | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (54 mil comentarios de Reddit) | Comentario → subconjunto de 27 emociones + `neutral` | RoBERTa-base |
| Extracción de habilidades | Etiquetado de secuencias (BIO) | [SkillSpan](https://huggingface.co/datasets/jjzha/skillspan) (11.5 mil oraciones de ofertas de empleo) | Oración → etiqueta `SKILL` / `KNOWLEDGE` / `O` por palabra | DeBERTa-v3-base |
| Desintoxicación de texto | Sequence-to-sequence | [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) (19.7 mil pares) | Comentario tóxico → versión respetuosa con el mismo sentido | BART-base |

## Metodología

1. **Descripción y análisis exploratorio** de cada base de datos: ejemplos, distribución de etiquetas, longitudes en
   tokens y particularidades. Por ejemplo, el 16 % de los comentarios de GoEmotions tiene varias emociones, y el 63 %
   de las entidades de prueba de SkillSpan no aparece en entrenamiento.
2. **Línea base clásica** para cada problema, que sirve de punto de comparación: TF-IDF con regresión logística, un
   diccionario de entidades vistas en entrenamiento y la eliminación de palabras de un léxico ofensivo.
3. **Búsqueda de hiperparámetros sobre una muestra**: *learning rate* entre `1e-5` y `1e-4`, y *batch size* de 16 y 32
   en emociones y desintoxicación. En SkillSpan, en lugar del *batch size*, se varió el modelo pre-entrenado (BERT,
   JobBERT o DeBERTa-v3).
4. **Entrenamiento final** con la data completa y la mejor configuración, en precisión mixta (fp16), conservando el
   mejor *checkpoint* según validación.
5. **Evaluación en el conjunto de prueba**, que no se usó para ninguna decisión: los hiperparámetros, los umbrales de
   decisión y los *checkpoints* se eligieron con el conjunto de validación. En ParaDetox, que se distribuye en un solo
   bloque, la partición se hizo agrupando por comentario, para que ninguno quede a la vez en entrenamiento y en prueba.

## Resultados en el conjunto de prueba

| Tarea | Métrica | Línea base clásica | Modelo fine-tuned | Mejor configuración | Entrenamiento final |
| :--- | :--- | :-: | :-: | :--- | :-: |
| Emociones | Macro-F1 | 0.420 | **0.523** | lr `5e-5`, batch 16, 4 épocas, un umbral por emoción | 6.4 min |
| Habilidades | F1 por entidad | 0.184 | **0.628** | DeBERTa-v3, lr `5e-5`, batch 16, 6 épocas (mejor *checkpoint*: época 5) | 2.5 min |
| Desintoxicación | BLEU | 63.4 | **66.4** | lr `1e-4`, batch 16, 4 épocas (mejor *checkpoint*: época 1) | 5.5 min |

Métricas complementarias:

- **Emociones:** Micro-F1 de 0.603, frente a 0.554 de la línea base. Las diferencias son grandes en emociones que
  dependen del contexto, como *curiosity* (0.55 vs. 0.37), *confusion* (0.49 vs. 0.33) y *disappointment* (0.34 vs. 0.15).
- **Habilidades:** F1 de 0.538 en `SKILL` y de 0.709 en `KNOWLEDGE`. En las entidades que no aparecen en entrenamiento, el
  modelo recupera el 53 % de las habilidades y el 65 % de los conocimientos; el diccionario no recupera ninguna.
- **Desintoxicación:** el 86.9 % de las salidas es no tóxico según un clasificador independiente (línea base: 85.8 %;
  reescrituras humanas: 94.3 %), con un chrF de 80.4.

Los tiempos corresponden a una GPU NVIDIA RTX 5070 Ti; en una T4 de Colab el entrenamiento es más lento.

## Conclusiones

- El *learning rate* fue el hiperparámetro más determinante. En GoEmotions, con batch 32 y valores de `1e-5` o
  `2e-5`, el modelo no aprendió nada en la muestra (F1 = 0) y se quedó prediciendo "ninguna emoción".
- En la extracción de habilidades, DeBERTa-v3 superó a BERT y a JobBERT con todos los *learning rates* probados: la
  arquitectura aportó más que el pre-entrenamiento en el dominio de las ofertas de empleo.
- Ajustar un umbral por emoción en validación subió el Macro-F1 de 0.509 (con un umbral global de 0.35) a 0.523, porque
  un único umbral no sirve a la vez para las emociones frecuentes y para las poco frecuentes.
- En la desintoxicación, eliminar palabras ofensivas resultó una línea base competitiva, porque muchas reescrituras
  humanas son eliminaciones. La ventaja del modelo está en los casos que requieren reformular.

## Limitaciones

- **Emociones poco frecuentes.** Emociones con muy pocos ejemplos reciben probabilidades bajas incluso cuando son
  evidentes. En *"My dog passed away this morning"* el modelo da 0.95 a *sadness* y solo 0.07 a *grief*, que se detecta
  únicamente porque su umbral ajustado es muy bajo.
- **Límites de las habilidades.** El modelo tiende a extender o fragmentar los spans largos. En *"solving business
  problems through innovation and engineering practices"* marca toda la frase como habilidad, cuando la anotación
  separa la habilidad (*solving business problems*) del conocimiento (*innovation and engineering practices*).
- **Desintoxicación incompleta.** Algunas salidas copian la entrada o solo la suavizan en parte: *"…a pathological liar
  like trump"* se mantiene igual, y *"…country of losers and abusers"* pasa a *"…country of people and abusers"*.
- **Comparabilidad.** La partición de ParaDetox es propia y cada configuración se entrenó con una sola semilla, por lo que
  diferencias menores a ~0.005 entre configuraciones no son concluyentes.

## Modelos publicados

| Tarea | Modelo en Hugging Face |
| :--- | :--- |
| Emociones | [`jeffespejo/roberta-base-goemotions-udea`](https://huggingface.co/jeffespejo/roberta-base-goemotions-udea) (incluye `thresholds.json` con el umbral de cada emoción) |
| Habilidades | [`jeffespejo/deberta-v3-base-skillspan-udea`](https://huggingface.co/jeffespejo/deberta-v3-base-skillspan-udea) |
| Desintoxicación | [`jeffespejo/bart-base-paradetox-udea`](https://huggingface.co/jeffespejo/bart-base-paradetox-udea) |

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

habilidades = pipeline("token-classification", model="jeffespejo/deberta-v3-base-skillspan-udea",
                       aggregation_strategy="simple")
print(habilidades("You will design and deploy NLP models. Knowledge of Docker and AWS is a plus."))

tokenizer = AutoTokenizer.from_pretrained("jeffespejo/bart-base-paradetox-udea")
detox = AutoModelForSeq2SeqLM.from_pretrained("jeffespejo/bart-base-paradetox-udea")
inputs = tokenizer("The referee was a complete idiot and ruined the damn game.", return_tensors="pt")
print(tokenizer.decode(detox.generate(**inputs, num_beams=4, max_length=128)[0], skip_special_tokens=True))
```

El código funciona con `transformers` 4.46 o superior, incluida la versión 5. El modelo de emociones es multi-etiqueta
y usa un umbral por emoción; la ficha del modelo explica cómo cargarlo.

## Ejecución

El notebook está pensado para Google Colab con GPU T4: la primera celda instala las dependencias que falten. Para
ejecutarlo localmente se requiere Python 3.10+ y los paquetes de [`requirements.txt`](./requirements.txt). Con la
variable de entorno `SMOKE=1` se ejecuta una versión reducida en pocos minutos, útil para verificar el entorno.

## Referencias

- Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL.
- Zhang, Jensen, Sonniks y Plank (2022). *SkillSpan: Hard and Soft Skill Extraction from English Job Postings*. NAACL.
- Logacheva et al. (2022). *ParaDetox: Detoxification with Parallel Data*. ACL.
- Liu et al. (2019), *RoBERTa*; He et al. (2021), *DeBERTaV3*; Lewis et al. (2020), *BART*.
- Hugging Face LLM Course: <https://huggingface.co/learn/llm-course>.

---

Proyecto 2 del curso *Fundamentos de Procesamiento de Lenguaje Natural*, Ingeniería de Sistemas, Universidad de
Antioquia (Prof. Antonio Jesús Tamayo Herrera, Ph.D.). Autor: Jefferson Espejo.
