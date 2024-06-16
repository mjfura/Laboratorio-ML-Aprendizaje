# Instalar faraway
install.packages("faraway")
install.packages("glmnet")
# Cargar faraway
library(faraway)
library(glmnet)

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

# 2. SELECCION DE VARIABLES
# Definir datos de entrenamiento y de prueba
set.seed(123)
num_observaciones <- nrow(chicago)
train_index <- sample(1:num_observaciones, 0.9*num_observaciones)
train_data <- chicago[train_index,]
test_data <- chicago[-train_index,]
print(nrow(test_data))
print(nrow(train_data))
# Modelo usando todas las variables
modelo_completo <- lm(involact ~ ., data = train_data)
summary(modelo_completo)
# Criterio de Información de Akaike (AIC)
modelo_aic <- step(lm(involact ~ ., data = train_data), direction = "both", trace = 1000)
summary(model_aic)
# Criterio de Información Bayesiano (BIC)
modelo_bic <- step(lm(involact ~ ., data = train_data),direction = "both" ,trace = 1000, k = log(length(train_data$involact)))
summary(modelo_bic)
# Conclusiones:
# - En este caso vemos que usando los criterios de AIC y BIC se seleccionan las mismas variables aunque los valores estimados son minúsculamente diferentes.
# - Con el uso de los criterios AIC y BIC ambos me consideran solo 4 variables como significativas para la elaboración del modelo lineal: race, fire, theft y age.

# 3. Significancia del Modelo
anova(modelo_completo)
anova(modelo_aic)
anova(modelo_bic)
anova(modelo_completo, modelo_aic,modelo_bic)
# El modelo obtenido es significativo, es decir que las variables seleccionadas son significativas para explicar la variable involact.
# Existen 2 covariables que no son significativas: volact y income según los criterios de AIC y BIC.
# - En base solo a los criterios AIC y BIC, el modelo obtenido es el mismo, por lo que no habría problema si es que removemos las dos variables menos significativas.

# 4. MÉTODOS DE REGRESION CONTRAIDAS

# - REGRESION RIDGE
X <- model.matrix(involact ~ . - 1, data = train_data)
y <- train_data$involact
# Seleccionar el valor de lambda
ridge_model <- cv.glmnet(X, y, alpha = 0)
coef_ridge <- coef(ridge_model, s = "lambda.min")
print(coef_ridge)
# - REGRESION LASSO
lasso_model <- cv.glmnet(X, y, alpha = 1)
coef_lasso <- coef(lasso_model, s = "lambda.min")
print(coef_lasso)
# - REGRESION ELASTIC NET
cv_elastic_net <- cv.glmnet(X, y, alpha = 0.5) # alpha = 0.5 es un balance entre Lasso (1) y Ridge (0)
best_lambda <- cv_elastic_net$lambda.min # Obtener el mejor valor de lambda
elastic_net_model <- glmnet(X, y, alpha = 0.5, lambda = best_lambda)
coef_elastic_net <- coef(elastic_net_model)
print(coef_elastic_net)
# Elaboracion modelos lineales:
selected_vars_ridge <- rownames(coef_ridge)[coef_ridge[, 1] != 0]
selected_vars_lasso <- rownames(coef_lasso)[coef_lasso[, 1] != 0]
selected_vars_elastic_net <- rownames(coef_elastic_net)[coef_elastic_net[, 1] != 0]
formula_ridge <- as.formula(paste("involact ~", paste(selected_vars_ridge[-1], collapse = " + ")))
formula_lasso <- as.formula(paste("involact ~", paste(selected_vars_lasso[-1], collapse = " + ")))
formula_elastic_net <- as.formula(paste("involact ~", paste(selected_vars_elastic_net[-1], collapse = " + ")))

modelo_ridge <- lm(formula_ridge, data = train_data)
modelo_lasso <- lm(formula_lasso, data = train_data)
modelo_elastic_net <- lm(formula_elastic_net, data = train_data)
anova(modelo_completo,modelo_aic,modelo_ridge, modelo_lasso, modelo_elastic_net)
# Conclusiones:
# - Al realizar la compración con los modelos anteriores y estos creados por los métodos de regresión contraidas, vemos que la significancia entre los modelos de ridge, lasso y elastic net es prácticamente el mismo ajuste, sin embargo al mismo tiempo este varía un poco del modelo obtenido por el criterio AIC y es muy similar por no decir igual que el modelo completo, y esto es de esperarse, ya que ninguno de estos métodos de regresión contraidas ha removido variables del modelo completo.

# 5. PREDICCION INDIVIDUAL Y DE LA MEDIA
# Modelos lineales AIC y el completo
# Predicción individual
# Predicción de la media
# Modelos Ridge, Lasso y Elastic Net
# Predicción individual

# Predicción de la media