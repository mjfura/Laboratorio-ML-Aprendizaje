# Instalar faraway
install.packages("faraway")
# Cargar faraway
library(faraway)
# Utilizar datos chicago
data(chicago)
?faraway::chicago

# 1. ANALISIS DESCRIPTIVO de las variables de la base de datos. Incluir indicadores y gráficos.
# Columnas
names(chicago)
# Verificar la estructura de la base de datos
str(chicago)
# Verificar las primeras observaciones
head(chicago)
# Verificar las últimas observaciones
tail(chicago)
# Indicadores de las variables
summary(chicago)
# Podemos ver que la mayoría de variables del dataset contienen algunos outliers.
# Gráficos
hist(chicago$involact)
# En este histograma podemos ver que la variable Y involact tiene una mayor concentración en el rango de 0 a 0.5 y a medida que va aumentando el valor de la variable, la frecuencia disminuye.
plot(density(chicago$involact))
boxplot(chicago$involact)
boxplot(chicago$race)
boxplot(chicago$fire)
boxplot(chicago$theft)
boxplot(chicago$age)
boxplot(chicago$volact)
boxplot(chicago$income)
pairs(chicago)
nrow(chicago)

# 2. SELECCION DE VARIABLES
# Modelo usando todas las variables
modelo_completo <- lm(involact ~ ., data = chicago)
summary(modelo_completo)
# Criterio de Información de Akaike (AIC)
modelo_aic <- step(lm(involact ~ ., data = chicago), direction = "both", trace = 1000)
summary(model_aic)
# Criterio de Información Bayesiano (BIC)
modelo_bic <- step(lm(involact ~ ., data = chicago),direction = "both" ,trace = 1000, k = log(length(chicago$involact)))
summary(modelo_bic)
# En este caso vemos que usando los criterios de AIC y BIC se seleccionan las mismas variables

# 3. Significancia del Modelo
