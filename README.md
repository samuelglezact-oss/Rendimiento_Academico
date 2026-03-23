## Rendimiento Academico

### Problema Hipótesis Variables 

##### Problema 
<p>
El análisis del rendimiento académico es el tema que optamos para su estudio ya que buscamos comprender los factores que influyen en el desempeño de los estudiantes que se encuentran cursando la licenciatura. Bajo este contexto, usaremos el modelo de regresión múltiple que nos permite identificar relaciones significativas entre diversas variables explicativas y una variable dependiente que represente el nivel de desempeño académico. 
Durante el presente proyecto queremos buscar y analizar la existencia de relaciones entre un conjunto de variables obtenidas a partir de una base de datos construida mediante un formulario digital, y el rendimiento académico de los estudiantes, definido como “promedio”. 
<p>

##### Hipotesis 
<p>
La hipótesis general que queremos probar es que todas las variables que consideramos explican el modelo.
<p>
$$
H_0 : \beta_1 = \beta_2 = \cdots = \beta_k = 0
$$
$$
H_1 : \exists \beta_i \neq 0
$$

<p>
Por otro lado, las hipótesis individuales que deseamos probar sobre el efecto de las variables explicativas en el rendimiento académico de los estudiantes son las siguientes: 
</p>

•	Se espera que las horas de sueño, las horas de estudio y el número de comidas tengan un efecto positivo sobre el promedio académico ya que las 3 se encuentran relacionadas con el bienestar y esfuerzo de los estudiantes. 

•	Esperamos que las variables como el nivel de estrés, tiempo de traslado, horas dedicadas a redes sociales y trabajo tengan un efecto negativo ya que reducen el tiempo disponible para estudiar. 

•	También, esperamos que las materias inscritas tengan un efecto marginal negativo ya que, a mayor carga académica, menos atención se le puede dedicar a cada materia.

•	Por último, las variables relacionadas con el tipo de transporte se incluyen como variables categóricas, de esta forma, su efecto dependerá de las condiciones específicas de traslado. 


##### Variables 
###### Justificacion de variables 
<p>
En primer lugar, tenemos que las horas de sueño son un factor fundamental para el adecuado funcionamiento cognitivo. Existen diversos artículos como el “Sueño y rendimiento académico en estudiantes universitarios: una revisión sistemática” que demuestran que un descanso adecuado mejora la memoria, la concentración y la capacidad de aprendizaje, lo que se refleja en un mejor rendimiento académico.

Igualmente, las horas de estudio están directamente relacionadas ya que estudiantes con buenos hábitos de estudio mejoran notablemente su desempeño académico. La implementación de rutinas de estudio, planificación adecuada de horarios y la utilización de técnicas de organización permite a los alumnos establecer su tiempo de manera eficaz, a través de ello sus calificaciones son satisfactorias, según el artículo “Hábitos de estudio y el desempeño académico de estudiantes”

Por otro lado, consideramos que el tiempo destinado a actividades no académicas como el uso de las redes sociales, puede afectar de forma negativa el rendimiento. El artículo “Efectos del uso de redes sociales en el desempeño académico en estudiantes universitarios” confirma que existe una correlación negativa entre el uso de redes sociales y el desempeño académico de los estudiantes universitarios, además sugiere que cuando los estudiantes no regulan su tiempo en redes, pueden experimentar una disminución en su rendimiento académico debido a la distracción y la falta de gestión de tiempo. 

Actualmente, el nivel de estrés es otro factor relevante, ya que influye en el bienestar emocional y en la capacidad de concentración ya que altos niveles de estrés pueden deteriorar el desempeño académico.

La carga académica, descrita por el número de materias inscritas puede afectar el rendimiento, ya que a mayor número de materias inscritas implica menor tiempo para dedicarle a cada materia y en el esfuerzo del estudiante tiene que ser más, lo que ocasiona un rendimiento académico decreciente.

Similarmente, que el estudiante esté trabajando puede afectar negativamente su rendimiento en la universidad porque implica que destine parte de su tiempo en actividades no académicas y deje menos para hacer tareas, estudiar o hacer notas por su cuenta. 

El número de comidas diarias se relaciona con si el estudiante tiene una buena alimentación o no ya que según el artículo científico “Impacto de la alimentación en el rendimiento académico” afirma que con una alimentación balanceada los estudiantes obtuvieron mejores calificaciones y prestaron un mayor nivel de atención en clase. Asimismo, demostraron que el no desayunar y consumir frecuentemente alimentos ultraprocesados, se relaciona con un menor rendimiento académico. 

Finalmente, el tiempo de traslado representa un costo en términos de tiempo y energía puesto que si el trayecto de su casa a la universidad es largo, puede afectar significativamente su bienestar y rendimiento porque va a estar más cansado y va a tener menos tiempo para cumplir con sus obligaciones escolares. 
<p>

### Cómo reproducir (comandos).

```
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
```


### Roles y estructura del repositorio.

<p>
En el desarrollo del proyecto se asignaron roles específicos a cada integrante con el fin de asegurar una adecuada organización del trabajo, así como la correcta ejecución metodológica, técnica y analítica del modelo de regresión lineal múltiple. A continuación, se describen las responsabilidades de cada miembro del equipo:
<p>

- ##### Líder del proyecto: Alejandra Sánchez Luna
> Es responsable de la dirección general del proyecto y de la coherencia metodológica. Entre sus funciones se encuentran la definición de la pregunta de investigación, la formulación de las hipótesis con sustento teórico y la especificación del modelo base. Asimismo, coordina al equipo, da seguimiento al cronograma de trabajo y supervisa que los resultados obtenidos tengan sentido económico o financiero. También valida la integración final del documento y la presentación.
- ##### Data Engineer: Ismael Morales Solís
>Se encarga de la adquisición, limpieza y preparación de los datos. Sus funciones incluyen la búsqueda y selección de fuentes de información confiables, la descarga y estructuración de la base de datos, el tratamiento de valores faltantes y la detección de inconsistencias. Además, identifica outliers preliminares, elabora el diccionario de variables (definición, tipo y unidad de medida) y genera el conjunto de datos final listo para el modelado, documentando todo el proceso.
- ##### Modelador: Luis A. Mariscal Armenta
>Es responsable de la construcción y estimación de los modelos econométricos. Sus tareas incluyen la implementación del modelo de regresión lineal múltiple (MCO) en Python, la selección de variables explicativas y el ajuste de la especificación del modelo. También desarrolla modelos alternativos (transformaciones logarítmicas, polinomios, variables dummy o interacciones), interpreta los coeficientes estimados y compara modelos mediante métricas como R², RMSE y MAE.
- ##### Validación y diagnóstico: Maryangel Vázquez Alavez
>Tiene a su cargo la evaluación estadística del modelo. Realiza el análisis de multicolinealidad mediante el cálculo del VIF y matrices de correlación, así como pruebas de heterocedasticidad como Breusch-Pagan o White. Además, evalúa la correcta especificación del modelo con la prueba de Ramsey RESET y detecta observaciones influyentes mediante indicadores como Cook’s Distance, leverage y DFBetas. Propone ajustes como el uso de errores robustos y analiza la validez general del modelo.
- ##### Visualización y narrativa: Ariadna Díaz Durán
>Es responsable de la presentación visual y la comunicación de los resultados. Elabora gráficos del análisis exploratorio de datos (distribuciones, correlaciones y outliers), diseña tablas claras de resultados y traduce los hallazgos técnicos a un lenguaje accesible. También desarrolla la presentación en diapositivas, construye la narrativa del proyecto y apoya en la redacción del documento final, así como en la preparación del contenido visual para el video.
- ##### Reproducibilidad y QA: Samuel González Islas
> Se encarga de garantizar la correcta estructura y funcionamiento técnico del proyecto. Organiza el repositorio (carpetas, datos y notebooks), elabora y mantiene el archivo README, y genera el archivo de dependencias (requirements.txt). Además, asegura que el proyecto sea reproducible, verificando que el código se ejecute sin errores en distintos entornos. También lleva el control de versiones mediante Git, revisa la consistencia y claridad del código, y establece un flujo de ejecución ordenado para facilitar su uso.

### Dependencias (pip install -r requirements.txt).

<p>
Para la correcta ejecución del proyecto, es necesario instalar previamente las librerías utilizadas en el desarrollo del análisis y la estimación del modelo. Estas dependencias se encuentran especificadas en el archivo requirements.txt, lo que permite su instalación automática mediante el uso del siguiente comando en la terminal:
pip install -r requirements.txt

Las principales librerías empleadas en el proyecto son las siguientes:
<p>

- pandas: utilizada para la manipulación, limpieza y estructuración de los datos.
- numpy: empleada para operaciones numéricas y manejo de arreglos.
- scipy: utilizada para funciones estadísticas y pruebas complementarias.
- statsmodels: empleada para la estimación del modelo de regresión lineal múltiple y la realización de pruebas econométricas.
- scikit-learn (sklearn.metrics): utilizada para la evaluación del modelo mediante métricas como RMSE y MAE.
- matplotlib: utilizada para la generación de gráficos y visualización de resultados. 

<p>
El uso del archivo requirements.txt garantiza la reproducibilidad del proyecto, permitiendo que cualquier usuario pueda instalar las mismas versiones de las librerías y ejecutar el código sin inconvenientes.
<p
