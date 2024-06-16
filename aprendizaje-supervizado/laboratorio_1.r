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
# Conclusiones:
# - Con el uso de los criterios AIC y BIC ambos me consideran solo 4 variables como significativas para la elaboración del modelo lineal

# 3. Significancia del Modelo
anova(modelo_completo)
anova(modelo_aic)
anova(modelo_bic)
anova(modelo_completo, modelo_aic)
# Ambos modelos son significativos
# Existen 2 covariables que no son significativas: volact y income
# El modelo lineal generado ocupando solo las 4 variables significativas tiene un ajuste muy similar al modelo completo. Es decir remover las 2 variables menos significativas no genera una mejora significativa en el ajuste del modelo.


# 4. MÉTODOS DE REGRESION CONTRAIDAS

# - REGRESION RIDGE
# Seleccionar el valor de lambda
lambda <- seq(0, 1, 0.1)
ridge <- cv.glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 0, lambda = lambda)
plot(ridge)
best_lambda <- ridge$lambda.min
best_lambda
ridge_model <- glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 0, lambda = best_lambda)
coef(ridge_model)
# - REGRESION LASSO
lasso <- cv.glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 1, lambda = lambda)
plot(lasso)
best_lambda <- lasso$lambda.min
best_lambda
lasso_model <- glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 1, lambda = best_lambda)
coef(lasso_model)
# - REGRESION ELASTIC NET
elastic_net <- cv.glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 0.5, lambda = lambda)
plot(elastic_net)
best_lambda <- elastic_net$lambda.min
best_lambda
elastic_net_model <- glmnet(as.matrix(chicago[, -1]), chicago$involact, alpha = 0.5, lambda = best_lambda)
coef(elastic_net_model)

# 5. PREDICCION INDIVIDUAL Y DE LA MEDIA
# Modelos lineales AIC y el completo
# Predicción individual
# Predicción de la media
# Modelos Ridge, Lasso y Elastic Net
# Predicción individual
# Predicción de la media