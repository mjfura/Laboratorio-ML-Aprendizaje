# SOLUCION LABORATORIO 1
# Integrantes:
# - Marcos Esteban Reiman Durán
# - Marco Josué Fura Mendoza
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
# En este caso estamos dejando 5 observaciones para realizar las predicciones en el punto 5, el resto se está usando como data de entrenamiento.
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

#Comentarios de resultados****
#Residual standard error: 0.3391 on 35 degrees of freedom
#Multiple R-squared:  0.7669,    Adjusted R-squared:  0.727
# F-statistic: 19.2 on 6 and 35 DF,  p-value: 9.352e-10
#DETALLE
# Residual standard error: 0.3391 --> Esto mide el error estándar , mide la calida de los predictores del modelo
# Multiple R-squared:  0.7669 --> el 76.69%  de la variabilidad se explica por las variables independientes
# Adjusted R-squared:  0.727  --> es un 72.7% al ser más bajo que el anterior indica que hay algun variable predictora que no agregar mucha información




# Criterio de Información de Akaike (AIC)
modelo_aic <- step(lm(involact ~ ., data = train_data), direction = "both", trace = 1000)
summary(modelo_aic)
#Resultado *****
#Hace 3 iteraciones 
# Start:  AIC=-84.51  <-- con todas las variables  --> involact ~ race + fire + theft + age + volact + income
# Step:  AIC=-84.82   <-- quita la variable income --> involact ~ race + fire + theft + age + volact
# Step:  AIC=-85.27    <-- quita la variable volact --> involact ~ race + fire + theft + age
# las variables descartas (volact,income) en las iteraciones
# El resultado del modelo 
# Residual standard error: 0.3427 on 37 degrees of freedom
# Multiple R-squared:  0.7482,    Adjusted R-squared:  0.721
# F-statistic: 27.49 on 4 and 37 DF,  p-value: 1.23e-10
# Residual standard error: 0.3391 <-- el bajo error residual es del modelo completo






# Criterio de Información Bayesiano (BIC)
modelo_bic <- step(lm(involact ~ ., data = train_data),direction = "both" ,trace = 1000, k = log(length(train_data$involact)))
summary(modelo_bic)
# Conclusiones:
# - En este caso vemos que usando los criterios de AIC y BIC se seleccionan las mismas variables aunque los valores estimados son minúsculamente diferentes.
# - Con el uso de los criterios AIC y BIC ambos me consideran solo 4 variables como significativas para la elaboración del modelo lineal: race, fire, theft y age.

# 3. Significancia del Modelo
anova(modelo_completo, modelo_aic,modelo_bic)
# Ambos modelos son significativos
# Existen 2 covariables que no son significativas: volact y income
# El modelo lineal generado ocupando solo las 4 variables significativas tiene un ajuste muy similar al modelo completo. Es decir remover las 2 variables menos significativas no genera una mejora significativa en el ajuste del modelo.
# El modelo no mejora significativamente al quitar las variables no significativas.

# 4. MÉTODOS DE REGRESION CONTRAIDAS

# - REGRESION RIDGE

X <- model.matrix(involact ~ . - 1, data = train_data)
y <- train_data$involact
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
valores_prueba<-test_data[,colnames(test_data)!="involact"]
# Modelos lineales AIC y el completo
# Predicción individual
pred_individual_modelo_completo <- predict(modelo_completo, valores_prueba, interval = "prediction", level = 0.95)
pred_individual_modelo_aic <- predict(modelo_aic, valores_prueba, interval = "prediction", level = 0.95)
# Mostramos los valores individuales predichos con sus intervalos de confianza en un 95%
print(pred_individual_modelo_completo)
print(pred_individual_modelo_aic)
# Predicción de la media
pred_media_modelo_completo <- predict(modelo_completo, valores_prueba, interval = "confidence", level = 0.95)
pred_media_modelo_aic <- predict(modelo_aic, valores_prueba, interval = "confidence", level = 0.95)
# Mostramos los valores medios predichos con sus intervalos de confianza en un 95%
print(pred_media_modelo_completo)
print(pred_media_modelo_aic)
# Modelos Ridge, Lasso y Elastic Net
# Predicción individual
pred_individual_modelo_ridge <- predict(modelo_ridge, valores_prueba, interval = "prediction", level = 0.95)
pred_individual_modelo_lasso <- predict(modelo_lasso, valores_prueba, interval = "prediction", level = 0.95)
pred_individual_modelo_elastic_net <- predict(modelo_elastic_net, valores_prueba, interval = "prediction", level = 0.95)
# Mostramos los valores individuales predichos con sus intervalos de confianza en un 95%
print(pred_individual_modelo_ridge)
print(pred_individual_modelo_lasso)
print(pred_individual_modelo_elastic_net)
# Predicción de la media
pred_media_modelo_ridge <- predict(modelo_ridge, valores_prueba, interval = "confidence", level = 0.95)
pred_media_modelo_lasso <- predict(modelo_lasso, valores_prueba, interval = "confidence", level = 0.95)
pred_media_modelo_elastic_net <- predict(modelo_elastic_net, valores_prueba, interval = "confidence", level = 0.95)
# Mostramos los valores medios predichos con sus intervalos de confianza en un 95%
print(pred_media_modelo_ridge)
print(pred_media_modelo_lasso)
print(pred_media_modelo_elastic_net)
# En este caso sabemos que el mejor ajuste es el del modelo completo con un R2 ajustado de 0.727
# EL modelo AIC le sigue con un R2 ajustado de 0.7231
# Los modelos de regresión contraidas tienen un R2 ajustado de 0.727 los 3, por lo que tiene un ajuste muy similar por no decir igual al del modelo completo y eso también ocurre por el uso de las 6 variables en lugar de 4 como lo hace el modelo AIC.
