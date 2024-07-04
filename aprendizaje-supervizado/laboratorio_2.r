# CARGAR LIBRERIAS
install.packages("splitTools")
install.packages("rpart.plot")
install.packages("caret")
install.packages("randomForest")
library(faraway)
library(splitTools)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(pROC)


# Función para verificar e instalar paquetes si es necesario
verificar_paquete <- function(paquete) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

# Verificar e instalar los paquetes necesarios
paquetes <- c("splitTools", "rpart.plot", "caret", "randomForest", "ggplot2", "faraway", "pROC")

for (paquete in paquetes) {
  verificar_paquete(paquete)
}

# Cargar librerías necesarias
library(faraway)
library(splitTools)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

# Utilizar el conjunto de datos "chicago" disponible en la librería "faraway".
# Considerar Y = involact como variable respuesta, todas las demás serán variables explicativas.

data(chicago, package = "faraway")
head(chicago)

# Filtrar solo las columnas numéricas
chicago_numerico <- chicago[, sapply(chicago, is.numeric)]

# Guardar el boxplot en un archivo PNG con dimensiones específicas
png("boxplot.png", width = 1200, height = 800)

# Ajustar los márgenes para evitar el error "figure margins too large"
par(mar = c(5, 8, 4, 2) + 0.1)

# Crear boxplots para todas las columnas numéricas
boxplot(chicago_numerico,
        main = "Boxplots de todas las columnas numéricas",
        las = 2,
        col = rainbow(ncol(chicago_numerico))
)

# Cerrar el dispositivo gráfico
dev.off()

# Funciones para calcular medidas de desempeño
# Esta función calcula el Error Cuadrático Medio (MSE)
calcular_mse <- function(real, prediccion) {
  return(mean((real - prediccion)^2))
}

# Esta función calcula el Error Absoluto Medio (MAE)
calcular_mae <- function(real, prediccion) {
  return(mean(abs(real - prediccion)))
}

# Esta función calcula el Coeficiente de Determinación (R²)
calcular_r2 <- function(real, prediccion) {
  ss_res <- sum((real - prediccion)^2)
  ss_tot <- sum((real - mean(real))^2)
  return(1 - (ss_res / ss_tot))
}

# Función para realizar una separación aleatoria para validación cruzada simple
split_cv_simple <- function(dataset) {
  ind <- splitTools::partition(dataset$involact, p = c(0.5, 0.2, 0.3), type = "stratified")
  datos_entrenamiento <- dataset[ind$`1`, ]
  datos_validacion <- dataset[ind$`2`, ]
  datos_prueba <- dataset[ind$`3`, ]
  
  entren_x <- datos_entrenamiento[, -which(names(datos_entrenamiento) == "involact")]
  entren_y <- datos_entrenamiento$involact
  
  valid_x <- datos_validacion[, -which(names(datos_validacion) == "involact")]
  valid_y <- datos_validacion$involact
  
  prueba_x <- datos_prueba[, -which(names(datos_prueba) == "involact")]
  prueba_y <- datos_prueba$involact
  
  return(list(entren_x, entren_y, valid_x, valid_y, prueba_x, prueba_y))
}

# Ejecutar la función de separación de datos
result <- split_cv_simple(chicago)
entren_x <- result[[1]]
entren_y <- result[[2]]
valid_x <- result[[3]]
valid_y <- result[[4]]
prueba_x <- result[[5]]
prueba_y <- result[[6]]

# Entrenar modelos K-NN, Árboles de Regresión y Random Forest y evaluar el mejor modelo con la muestra de validación

# Entrenar modelo KNN
modelo_knn <- train(
  x = entren_x, y = entren_y,
  method = "knn"
)
prediccion_knn <- predict(modelo_knn, newdata = valid_x)
mse_knn <- calcular_mse(valid_y, prediccion_knn)
mae_knn <- calcular_mae(valid_y, prediccion_knn)
r2_knn <- calcular_r2(valid_y, prediccion_knn)
print(c(mse_knn, mae_knn, r2_knn))

# Entrenar modelo Árbol de Regresión
arbol_regresion <- rpart(involact ~ ., data = data.frame(entren_x, involact = entren_y), method = "anova")
rpart.plot(arbol_regresion)
predicciones_arbol <- predict(arbol_regresion, newdata = valid_x)
mse_arbol <- calcular_mse(valid_y, predicciones_arbol)
mae_arbol <- calcular_mae(valid_y, predicciones_arbol)
r2_arbol <- calcular_r2(valid_y, predicciones_arbol)
print(c(mse_arbol, mae_arbol, r2_arbol))

# Entrenar modelo Random Forest
modelo_rf <- randomForest(involact ~ ., data = data.frame(entren_x, involact = entren_y), ntree = 500)
predicciones_rf <- predict(modelo_rf, newdata = valid_x)
mse_rf <- calcular_mse(valid_y, predicciones_rf)
mae_rf <- calcular_mae(valid_y, predicciones_rf)
r2_rf <- calcular_r2(valid_y, predicciones_rf)
print(c(mse_rf, mae_rf, r2_rf))

# Seleccionar el mejor modelo según MSE
mejor_modelo <- ifelse(min(mse_knn, mse_arbol, mse_rf) == mse_knn, "KNN",
                       ifelse(min(mse_knn, mse_arbol, mse_rf) == mse_arbol, "Árbol de Regresión", "Random Forest"))
print(mejor_modelo)

# Evaluar el mejor modelo con la muestra de prueba
if (mejor_modelo == "KNN") {
  predicciones_prueba <- predict(modelo_knn, newdata = prueba_x)
} else if (mejor_modelo == "Árbol de Regresión") {
  predicciones_prueba <- predict(arbol_regresion, newdata = prueba_x)
} else {
  predicciones_prueba <- predict(modelo_rf, newdata = prueba_x)
}
mse_prueba <- calcular_mse(prueba_y, predicciones_prueba)
mae_prueba <- calcular_mae(prueba_y, predicciones_prueba)
r2_prueba <- calcular_r2(prueba_y, predicciones_prueba)
print(c(mse_prueba, mae_prueba, r2_prueba))

# Función para realizar una separación aleatoria para validación cruzada k-fold
split_cv_fold <- function(dataset, k_fold) {
  ind <- splitTools::partition(dataset$involact, p = c(0.7, 0.3), type = "stratified")
  datos_entrenamiento <- dataset[ind$`1`, ]
  datos_prueba <- dataset[ind$`2`, ]
  
  entren_x <- datos_entrenamiento[, -which(names(datos_entrenamiento) == "involact")]
  entren_y <- datos_entrenamiento$involact
  
  prueba_x <- datos_prueba[, -which(names(datos_prueba) == "involact")]
  prueba_y <- datos_prueba$involact
  
  control <- trainControl(method = "cv", number = k_fold, savePredictions = TRUE)
  
  return(list(entren_x, entren_y, prueba_x, prueba_y, control))
}

# Ejecutar la función de separación de datos k-fold
result <- split_cv_fold(chicago, 10)
entren_x <- result[[1]]
entren_y <- result[[2]]
prueba_x <- result[[3]]
prueba_y <- result[[4]]
control <- result[[5]]

# Entrenar modelos K-NN, Árboles de Regresión y Random Forest con validación cruzada k-fold

# Entrenar modelo KNN con validación cruzada k-fold
modelo_knn <- train(
  x = entren_x, y = entren_y,
  method = "knn",
  trControl = control
)
predicciones_knn <- modelo_knn$pred
mse_knn <- mean((predicciones_knn$obs - predicciones_knn$pred)^2)
mae_knn <- mean(abs(predicciones_knn$obs - predicciones_knn$pred))
r2_knn <- calcular_r2(predicciones_knn$obs, predicciones_knn$pred)
print(c(mse_knn, mae_knn, r2_knn))

# Entrenar modelo Árbol de Regresión con validación cruzada k-fold
arbol_regresion <- train(
  x = entren_x, y = entren_y,
  method = "rpart",
  trControl = control
)
predicciones_arbol <- arbol_regresion$pred
mse_arbol <- mean((predicciones_arbol$obs - predicciones_arbol$pred)^2)
mae_arbol <- mean(abs(predicciones_arbol$obs - predicciones_arbol$pred))
r2_arbol <- calcular_r2(predicciones_arbol$obs, predicciones_arbol$pred)
print(c(mse_arbol, mae_arbol, r2_arbol))

# Entrenar modelo Random Forest con validación cruzada k-fold
tuneGrid <- expand.grid(.mtry = 3)
modelo_rf <- train(
  x = entren_x, y = entren_y,
  method = "rf",
  trControl = control,
  tuneGrid = tuneGrid,
  ntree = 500
)
predicciones_rf <- modelo_rf$pred
mse_rf <- mean((predicciones_rf$obs - predicciones_rf$pred)^2)
mae_rf <- mean(abs(predicciones_rf$obs - predicciones_rf$pred))
r2_rf <- calcular_r2(predicciones_rf$obs, predicciones_rf$pred)
print(c(mse_rf, mae_rf, r2_rf))

# Seleccionar el mejor modelo según MSE en validación cruzada k-fold
mejor_modelo_kfold <- ifelse(min(mse_knn, mse_arbol, mse_rf) == mse_knn, "KNN",
                             ifelse(min(mse_knn, mse_arbol, mse_rf) == mse_arbol, "Árbol de Regresión", "Random Forest"))
print(mejor_modelo_kfold)

# Ajustar los parámetros con toda la muestra de entrenamiento y usar la muestra de prueba
if (mejor_modelo_kfold == "KNN") {
  predicciones_prueba_kfold <- predict(modelo_knn, newdata = prueba_x)
} else if (mejor_modelo_kfold == "Árbol de Regresión") {
  predicciones_prueba_kfold <- predict(arbol_regresion, newdata = prueba_x)
} else {
  predicciones_prueba_kfold <- predict(modelo_rf, newdata = prueba_x)
}
mse_prueba_kfold <- calcular_mse(prueba_y, predicciones_prueba_kfold)
mae_prueba_kfold <- calcular_mae(prueba_y, predicciones_prueba_kfold)
r2_prueba_kfold <- calcular_r2(prueba_y, predicciones_prueba_kfold)
print(c(mse_prueba_kfold, mae_prueba_kfold, r2_prueba_kfold))


# PREGUNTAS PARTE 02
# 1. De acuerdo al contexto del conjunto de datos,
# seleccione alguna de las medidas de desempeñno que considere apropiada.
# Explique la elección

?faraway::wbca

data(wbca)
head(wbca)
table(wbca$Class)
wbca_numeric <- wbca[, sapply(wbca, is.numeric)]

# Creando boxplots para todas las columnas numéricas
boxplot(wbca_numeric,
    main = "Boxplots de todas las columnas numéricas",
    las = 2,
    col = rainbow(ncol(wbca_numeric)) 
)
# Podemos ver que las clases están desbalanceadas
# ya que los casos benignos son casi el doble que los malignos
# por lo que usaremos como métrica de desempeño el F1 Score

calcular_f1_score <- function(matrix_confusion) {
    verdaderos_positivos <- matrix_confusion$table[1, 1]
    falsos_positivos <- matrix_confusion$table[1, 2]
    precision_matrix <- verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
    sensibilidad_matrix <- matrix_confusion$byClass["Sensitivity"]
    f1_score_matrix <- 2 * ((precision_matrix * sensibilidad_matrix) / (precision_matrix + sensibilidad_matrix))
    return(f1_score_matrix)
}
split_cv_simple <- function(dataset) {
    ind <- splitTools::partition(dataset$Class, p = c(0.5, 0.2, 0.3), type = "stratified")
    data_train <- dataset[ind$`1`, ]
    data_valid <- dataset[ind$`2`, ]
    data_test <- dataset[ind$`3`, ]

    train_x <- data_train[, 2:ncol(data_train)]
    train_y <- data_train[, 1]
    train_y_factor <- factor(train_y, levels = c(1, 0))

    valid_x <- data_valid[, 2:ncol(data_valid)]
    valid_y <- data_valid[, 1]
    valid_y_factor <- factor(valid_y, levels = c(1, 0))

    test_x <- data_test[, 2:ncol(data_test)]
    test_y <- data_test[, 1]
    test_y_factor <- factor(test_y, levels = c(1, 0))
    return(list(train_x, train_y, train_y_factor, valid_x, valid_y, valid_y_factor, test_x, test_y, test_y_factor))
}
split_cv_fold <- function(dataset,k_fold){
    ind <- splitTools::partition(dataset$Class, p = c(0.7, 0.3), type = "stratified")
    data_train <- dataset[ind$`1`, ]
    data_test <- dataset[ind$`2`, ]

    train_x <- data_train[, 2:ncol(data_train)]
    train_y <- data_train[, 1]
    train_y_factor <- factor(train_y, levels = c(1, 0))

    test_x <- data_test[, 2:ncol(data_test)]
    test_y <- data_test[, 1]
    test_y_factor <- factor(test_y, levels = c(1, 0))
    control <- trainControl(method = "cv", number = k_fold, savePredictions = TRUE)

    return(list(train_x, train_y, train_y_factor, test_x, test_y, test_y_factor,control))

}
# 2.Realizar una separación aleatoria de la data original,
# para realizar una validación
# cruzada simple, donde la muestra de entrenamiento considera el 50%,
# la muestra de validación
# un 20% y la muestra test el 30%. A partir de estas:

set.seed(0)
result <- split_cv_simple(wbca)
train_x <- result[[1]]
train_y <- result[[2]]
train_y_factor <- result[[3]]
valid_x <- result[[4]]
valid_y <- result[[5]]
valid_y_factor <- result[[6]]
test_x <- result[[7]]
test_y <- result[[8]]
test_y_factor <- result[[9]]
# 2.a. Entrenar con los modelos K-NN,
# Arboles de Regresión y Random Forest utilizando la muestra de entrenamiento.
# A partir de la medida de desempeño escogida anteriormente, ¿cuál
# de los métodos tiene el mejor resultado, según la muestra de validación?

# Modelo KNN
modelo_knn <- train(
    x = train_x, y = train_y_factor,
    method = "knn"
)
prediccion_knn <- predict(modelo_knn, newdata = valid_x)
matrix_confusion <- confusionMatrix(prediccion_knn, valid_y_factor)
f1_score_knn <- calcular_f1_score(matrix_confusion)
print(f1_score_knn)
# 0.9666667
# Modelo Arbol de Clasificacion
data_train_factor <- cbind(train_x,Class=train_y_factor)
arbol_decision <- rpart(Class ~ ., data = data_train_factor,method = "class")
rpart.plot(arbol_decision)
predictions <- predict(arbol_decision, valid_x, type = "class")
matrix_confusion <- confusionMatrix(predictions, valid_y_factor)
f1_score_arbol <- calcular_f1_score(matrix_confusion)
print(f1_score_arbol)
# 0.9411765
# Modelo Random Forest
rf_model <- randomForest(Class ~ ., data = data_train_factor, ntree = 500, mtry = 3, importance = TRUE)
predictions <- predict(rf_model, valid_x)
matrix_confusion <- confusionMatrix(predictions, valid_y_factor)
f1_score_rf <- calcular_f1_score(matrix_confusion)
print(f1_score_rf)
# 0.9662921
# Según las validaciones hechas con la métrica f1_score 
# tanto el modelo KNN como el random forest tienen los valores más altos

# 2.b. Testear Modelo escogido
predictions_test <- predict(modelo_knn, test_x)
matrix_confusion <- confusionMatrix(predictions_test, test_y_factor)
f1_score_test <- calcular_f1_score(matrix_confusion)
print(f1_score_test)
# 0.9847328
# Hemos obtenido un score incluso mayor que en el entrenamiento y la validación
# por lo que podemos decir que el modelo generaliza bien

# 3.
result <- split_cv_fold(wbca, 10)
train_x <- result[[1]]
train_y <- result[[2]]
train_y_factor <- result[[3]]
test_x <- result[[4]]
test_y <- result[[5]]
test_y_factor <- result[[6]]
control <- result[[7]]
# 3.a
# K-FOLD : 10
# KNN
modelo_knn <- train(
    x = train_x, y = train_y_factor,
    method = "knn",
    trControl = control
)
predicciones_knn <- modelo_knn$pred
matrix_confusion_knn <- confusionMatrix(predicciones_knn$pred, predicciones_knn$obs)
f1_score_knn <- calcular_f1_score(matrix_confusion_knn)
print(f1_score_knn)
# 0.9823812

# Arboldes de clasificacion
arbol_decision <- train(
    x = train_x, y = train_y_factor,
    method = "rpart",
    trControl = control
)
predicciones_arbol <- arbol_decision$pred
matrix_confusion_arbol <- confusionMatrix(predicciones_arbol$pred, predicciones_arbol$obs)
f1_score_arbol <- calcular_f1_score(matrix_confusion_arbol)
print(f1_score_arbol)
# 0.91280052

# Random Forest
tuneGrid <- expand.grid(.mtry = 3)

modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    trControl = control,
    tuneGrid = tuneGrid,
    ntree = 500
)
predicciones_rf <- modelo_rf$pred
matrix_confusion_rf <- confusionMatrix(predicciones_rf$pred, predicciones_rf$obs)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.983871

# DE LOS 3 TIPOS DE MODELOS EL QUE OBTUVO EL MEJOR F1 SCORE
# FUE EL RANDOM FOREST

# K-FOLD : 15
control_15 <- trainControl(method = "cv", number = 15, savePredictions = TRUE)

# KNN
modelo_knn <- train(
    x = train_x, y = train_y_factor,
    method = "knn",
    trControl = control_15
)
predicciones_knn <- modelo_knn$pred
matrix_confusion_knn <- confusionMatrix(predicciones_knn$pred, predicciones_knn$obs)
f1_score_knn <- calcular_f1_score(matrix_confusion_knn)
print(f1_score_knn)
# 0.9817987

# Arboldes de clasificacion
arbol_decision <- train(
    x = train_x, y = train_y_factor,
    method = "rpart",
    trControl = control_15
)
predicciones_arbol <- arbol_decision$pred
matrix_confusion_arbol <- confusionMatrix(predicciones_arbol$pred, predicciones_arbol$obs)
f1_score_arbol <- calcular_f1_score(matrix_confusion_arbol)
print(f1_score_arbol)
# 0.9312821

# Random Forest
tuneGrid <- expand.grid(.mtry = 3)
modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    trControl = control_15,
    tuneGrid = tuneGrid,
    ntree = 500
)
predicciones_rf <- modelo_rf$pred
matrix_confusion_rf <- confusionMatrix(predicciones_rf$pred, predicciones_rf$obs)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.983871

# LUEGO DE VALIDAR CON UN K FOLD DE 15 SE VE
# QUE EL F1 SCORE DE LOS 3 MODELOS CON RESPECTO AL K FOLD 10
# SE VEN MUY SIMILARES, TIENDENE A VARIAR MUY POCO
# PERO DE LOS 3 MODELOS QUIEN SIGUE TENIENDO UN MEJOR F1 SCORE
# ES EL RANDOM FOREST

# 3.b
tuneGrid <- expand.grid(.mtry = 3)

modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    tuneGrid = tuneGrid,
    ntree = 500
)
predicciones_rf <- predict(modelo_rf, test_x)
matrix_confusion_rf <- confusionMatrix(predicciones_rf, test_y_factor)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.9655172

# SE OBSERVA UN F1 SCORE QUE CONDICE CON LO ESPERADO
# RESULTANDO CERCANO AL DE EL ENTRENAMIENTO
# POR LO QUE PODEMOS DECIR QUE EL MODELO GENERALIZA BIEN

# 4.

# EN LA VALIDACION SIMPLE LOS MEJORES MODELOS FUERON EL KNN Y EL RANDOM FOREST
# PERO LIGERAMENTE SUPERIOR EL KNN CON UN F1 SCORE DE 0.9666667
# EN LA VALIDACION K FOLD DE 10 Y 15 EL MEJOR MODELO
# FUE EL RANDOM FOREST CON UN F1 SCORE DE 0.979066 y 0.9789984 respectivamente
# AL MOMENTO DE EVALUAR LA GENERALIZACION OBSERVAMOS
# 0.9847328 PARA UN MODELO KNN EN VALIDACION SIMPLE
# 0.9847328 PARA UN MODELO RANDOM FOREST K FOLD : 10
# PODEMOS CONCLUIR QUE PARA LA VALIDACION UNA METODLOGIA K FOLD RESULTA MEJOR QUE UNA VALIDACION SIMPLE
# Y EL RESULTADO DEL MODELO A ESCOGER POR MINIMAS DIFERENCIAS VARIAN EN EL PRIMERO EL KNN Y EN EL SEGUNDO EL RANDOM FOREST
# POR OTRO LADO AMBAS METODLOGIAS AL EVALUAR LA GENERALIZACION DEL MODELO CONVERGEN EN UN MISMO VALOR


#PREGUNTAS ADICIONALES:
# 1.
# 2.
result <- split_cv_simple(wbca)
train_x <- result[[1]]
train_y <- result[[2]]
train_y_factor <- result[[3]]
valid_x <- result[[4]]
valid_y <- result[[5]]
valid_y_factor <- result[[6]]
test_x <- result[[7]]
test_y <- result[[8]]
test_y_factor <- result[[9]]

calcular_auc <- function(valid_y_factor, prediccion_knn) {
    roc_obj <- roc(valid_y_factor, as.numeric(prediccion_knn))
    roc_auc <- auc(roc_obj)
    return(roc_auc)
}
# KNN Validacion Cruzada Simple
modelo_knn <- train(
    x = train_x, y = train_y_factor,
    method = "knn"
)
prediccion_knn <- predict(modelo_knn, newdata = test_x)
roc_auc <- calcular_auc(test_y_factor, prediccion_knn)
print(roc_auc)
# 0.9887

# Random Forest Validacion Cruzada K - Fold 10
result <- split_cv_fold(wbca, 10)
train_x <- result[[1]]
train_y <- result[[2]]
train_y_factor <- result[[3]]
test_x <- result[[4]]
test_y <- result[[5]]
test_y_factor <- result[[6]]
control <- result[[7]]
tuneGrid <- expand.grid(.mtry = 3)

modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    trControl = control,
    tuneGrid = tuneGrid,
    ntree = 500
)
predicciones_rf <- predict(modelo_rf, test_x)
roc_auc <- calcular_auc(test_y_factor, predicciones_rf)
print(roc_auc)
# 0.9647

# COMO METRICA DE RENDIMIENTO ALTERNATIVA USAREMOS EL AUC DE LA CURVA ROC
# PARA EL MODELO KNN EN VALIDACION CRUZADA SIMPLE OBTUVIMOS UN AUC DE 0.9887
# PARA EL MODELO RANDOM FOREST EN VALIDACION CRUZADA K FOLD 10 OBTUVIMOS UN AUC DE 0.9647
# POR LO QUE PODEMOS CONCLUIR QUE EL MODELO KNN ES EL QUE MEJOR SE AJUSTA A LOS DATOS SEGUN ESTA NUEVA METRICA