# LABORATORIO 02
# -----
# INTEGRANTES:
# - Marcos Esteban Reiman Durán
# - Marco Josué Fura Mendoza

# Instalar librerias
install.packages("splitTools")
install.packages("rpart.plot")
install.packages("caret")
install.packages("randomForest")
# !Instalar librerias

# Cargar librerias
library(faraway)
library(caret)
library(dplyr)
library(class)
library(rpart)
library(randomForest)
library(ggplot2)
library(pROC)
# !Cargar librerias
set.seed(0)

# Cargar datos
data(chicago, package = "faraway")
chicago
# Cambiamos los nombres de las columnas a español, nuestra variable objetivo Involact es ahora ActividadesInvoluntarias
names(chicago) <- c("Raza", "Incendios", "Robos", "Edad", "ActividadesVoluntarias", "ActividadesInvoluntarias", "Ingresos")
?chicago
# Preguntas parte 1
#1.- Seleccione alguna de las medidas de desempe˜ no que pueda ser utilizada en este conjunto
#de datos. Indique el criterio utilizado.

# Se evaluara con MSE : Penaliza más los errores grandes aunque no es sensible a los datos atipicos.

# Mostrar los primeros datos
head(chicago)
print(chicago)
nrow(chicago)

# Resumen de datos
summary(chicago)

# Correlación entre variables
cor(chicago)

# Modelo de regresión lineal
modelo <- lm(Ingresos ~ ., data = chicago)
summary(modelo)

# Histograma de ingresos
ggplot(chicago, aes(x = Ingresos)) + 
  geom_histogram(binwidth = 1000, fill = "blue", color = "red") +
  labs(title = "Histograma de Ingresos", x = "Ingresos", y = "Frecuencia")

# Diagrama de dispersión entre edad y ingresos
ggplot(chicago, aes(x = Edad, y = Ingresos)) + 
  geom_point() +
  labs(title = "Diagrama de Dispersión entre Edad e Ingresos", x = "Edad", y = "Ingresos")

# Escalar datos
boxplot(chicago)
scaled_data <- as.data.frame(scale(chicago))
boxplot(scaled_data)
# !Escalar datos

# (a) Entrenar con los modelos K-NN, Arboles de Regresion y Random Forest utilizando la muestra
# de entrenamiento. A partir de la medida de desempeno escogida anteriormente, ¿cual
# de los metodos tiene el mejor resultado, segun la muestra de validacion?

ind <- splitTools::partition(scaled_data$ActividadesInvoluntarias, p = c(0.5, 0.2, 0.3))
data_train <- scaled_data[ind$`1`, ]
data_valid <- scaled_data[ind$`2`, ]
data_test <- scaled_data[ind$`3`, ]

train_x <- data_train[, setdiff(seq_len(ncol(data_train)), 6)]
train_y <- data_train[, 6]
valid_x <- data_valid[, setdiff(seq_len(ncol(data_valid)), 6)]
valid_y <- data_valid[, 6]
test_x <- data_test[, setdiff(seq_len(ncol(data_test)), 6)]
test_y <- data_test[, 6]


# Entrenar los modelos 
# Modelo KNN
modelo_knn <- train(
    x = train_x, y = train_y,
    method = "knn"
)
modelo_knn
prediccion_knn <- predict(modelo_knn, newdata = valid_x)
mse_knn <- mean((prediccion_knn - valid_y)^2)
print(mse_knn)
# 0.2513297

# Modelo Arbol de Regresion
modelo_arbol <- rpart(ActividadesInvoluntarias ~ ., data = data_train, method = "anova")
predicciones_arbol <- predict(modelo_arbol, newdata = data_valid)
mse_arbol <- mean((predicciones_arbol - valid_y)^2)
print(mse_arbol)
# 0.5918206

# Modelo Random Forest
modelo_rf <- randomForest(ActividadesInvoluntarias ~ ., data = data_train, ntree = 500, mtry = 3, importance = TRUE)
predictions <- predict(modelo_rf, valid_x)
mse_rf <- mean((predictions - valid_y)^2)
print(mse_rf)
# 0.2878254

#  resultados
cat("K-NN MSE:", mse_knn, "\n")
cat("Árbol de Regresión MSE:", mse_arbol, "\n")
cat("Random Forest MSE:", mse_rf, "\n")

# Graficar los resultados
datos_MSE <- data.frame(
  Modelo = c("K-NN", "Árbol de Regresión", "Random Forest"),
  MSE = c(mse_knn, mse_arbol, mse_rf)
)

ggplot(datos_MSE, aes(x = Modelo, y = MSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = round(MSE, 2)), vjust = -0.5) +
  labs(title = "Comparación de MSE entre Modelos", x = "Modelo", y = "MSE") +
  theme_minimal()


#El modelo KNN es el que obtiene el mejor resultado con MSE más pequeño
# Predicciones en conjunto de prueba
modelo_knn <- train(
    x = rbind(train_x,valid_x), y = c(train_y,valid_y),
    method = "knn"
)
modelo_knn
prediccion_knn <- predict(modelo_knn, newdata = test_x)

# Calcular MSE en conjunto de prueba
mse_simple <- mean((prediccion_knn - test_y)^2)
print(mse_simple)

# Imprimir resultado final
cat("MSE en el conjunto de prueba:", mse_simple, "\n")


#(b) Para el modelo seleccionado en el paso anterior, utilice la muestra test para medir la calidad
#del ajuste. Comente los resultados
# MSE en el conjunto de prueba: 0.24574


#PREGUNTA 1: item 3
#Realizar una separacion aleatoria de la data original, para realizar una validacion
#cruzada k-fold (escoja un valor de k), donde la muestra de entrenamiento considera el 70% y
#la muestra test el 30% (solo para fines comparativos mantener la muestra test realizada en la
#separacion anterior). A partir de estas:



# Hacemos la separación de los datos en entrenamiento (70%) y prueba (30%)
ind <- splitTools::partition(scaled_data$ActividadesInvoluntarias, p = c(0.7, 0.3))
data_train <- scaled_data[ind$`1`, ]
data_test <- scaled_data[ind$`2`, ]

train_x <- data_train[, setdiff(seq_len(ncol(data_train)), 6)]
train_y <- data_train[, 6]

test_x <- data_test[, setdiff(seq_len(ncol(data_test)), 6)]
test_y <- data_test[, 6]
control <- trainControl(method = "cv", number = 5, savePredictions = TRUE)

# Entrenamiento de modelos
# KNN
modelo_knn <- train(
    x = train_x, y = train_y,
    method = "knn",
    trControl = control
)
predicciones_knn <- modelo_knn$pred
predicciones_knn
mse_knn <- mean((predicciones_knn$pred - predicciones_knn$obs)^2)
print(mse_knn)
# 0.5030658

# Arbol de regresion
arbol_regresion <- train(
    x = train_x, y = train_y,
    method = "rpart",
    trControl = control
)
predicciones_arbol <- arbol_regresion$pred
mse_arbol <- mean((predicciones_arbol$pred - predicciones_arbol$obs)^2)
print(mse_arbol)
# 0.8022707

# Random Forest

modelo_rf <- train(
    x = train_x, y = train_y,
    method = "rf",
    trControl = control
)
modelo_rf
predicciones_rf <- modelo_rf$pred
mse_rf <- mean((predicciones_rf$pred - predicciones_rf$obs)^2)
print(mse_rf)
# 0.4184307

  
# # Crear gráfico comparando los MSE de los modelos

df_mse <- data.frame(
  Modelo = c("K-NN", "Árbol de Regresión", "Random Forest"),
  MSE = c(mse_knn, mse_arbol, mse_rf)
)

ggplot(df_mse, aes(x = Modelo, y = MSE, fill = Modelo)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Comparación de MSE entre Modelos") +
  ylab("MSE") +
  xlab("Modelo")


# Probar nuevamente con k-fold seguerido de 10 
control_10 <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

# volver a probar los modelos
# KNN
modelo_knn <- train(
    x = train_x, y = train_y,
    method = "knn",
    trControl = control_10
)
predicciones_knn <- modelo_knn$pred
mse_knn <- mean((predicciones_knn$pred - predicciones_knn$obs)^2)
print(mse_knn)
# 0.4445108

# Arbol de regresion
arbol_decision <- train(
    x = train_x, y = train_y,
    method = "rpart",
    trControl = control_10
)
predicciones_arbol <- arbol_decision$pred
mse_arbol <- mean((predicciones_arbol$pred - predicciones_arbol$obs)^2)
print(mse_arbol)
# 1.058191

# Random Forest
modelo_rf <- train(
    x = train_x, y = train_y,
    method = "rf",
    trControl = control_10
)
modelo_rf
predicciones_rf <- modelo_rf$pred
mse_rf <- mean((predicciones_rf$pred - predicciones_rf$obs)^2)
print(mse_rf)
# 0.4445661

# Seleccionar el mejor modelo basado en la validación cruzada

# # Crear gráfico comparando los MSE de los modelos
df_mse_2 <- data.frame(
  Modelo = c("K-NN", "Árbol de Regresión", "Random Forest"),
  MSE = c(mse_knn, mse_arbol, mse_rf)
)

ggplot(df_mse_2, aes(x = Modelo, y = MSE, fill = Modelo)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Comparación de MSE entre Modelos") +
  ylab("MSE") +
  xlab("Modelo")

#En la primera iteración el KNN iba ganando pero luego de aumentar el K de 5 a 10 , random forest mejor el desempeño


# Ajustar el mejor modelo seleccionado (Random Forest) con toda la muestra de entrenamiento y medir con la muestra test

prediccion_prueba <- predict(modelo_rf, newdata = test_x)
mse_rf_k_fold <- mean((prediccion_prueba - test_y)^2)

print(paste("MSE del modelo Random Forest en la muestra de prueba:", mse_rf_k_fold))
# MSE del modelo Random Forest en la muestra de prueba: 0.13319362668877



df_mse_3 <- data.frame(
  Modelo = c( "Simple","k-fold"),
  MSE = c(mse_simple, mse_rf_k_fold)
)

ggplot(df_mse_3, aes(x = Modelo, y = MSE, fill = Modelo)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Comparación Tipos de Validación") +
  ylab("MSE") +
  xlab("Modelo")


# el resultado de K-fold es mucho mejor que el simple casi un 40% mejor.





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
matrix_confusion <- confusionMatrix(prediccion_knn, valid_y_factor,mode="everything")
f1_score_knn <- calcular_f1_score(matrix_confusion)
print(f1_score_knn)
# 0.9666667
# Modelo Arbol de Clasificacion
data_train_factor <- cbind(train_x,Class=train_y_factor)
control <- rpart.control(minsplit = 20, cp = 0.005)
arbol_decision <- rpart(Class ~ ., data = data_train_factor,
                        method = "class",
                        control = control)
rpart::plotcp(arbol_decision)
rpart::printcp(arbol_decision)
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
modelo_knn <- train(
    x = rbind(train_x,valid_x), y = c(train_y_factor,valid_y_factor),
    method = "knn"
)
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
# 0.9715818

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
# 0.9172853

# Random Forest
tuneGrid <- expand.grid(.mtry = 3)

modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    trControl = control,
    #tuneGrid = tuneGrid,
    #ntree = 500
)
predicciones_rf <- modelo_rf$pred
matrix_confusion_rf <- confusionMatrix(predicciones_rf$pred, predicciones_rf$obs)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.9746631

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
# 0.9700214

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
# 0.9020619

# Random Forest
tuneGrid <- expand.grid(.mtry = 3)
modelo_rf <- train(
    x = train_x, y = train_y_factor,
    method = "rf",
    trControl = control_15
    #tuneGrid = tuneGrid,
    #ntree = 500
)
modelo_rf
predicciones_rf <- modelo_rf$pred
matrix_confusion_rf <- confusionMatrix(predicciones_rf$pred, predicciones_rf$obs)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.9757674

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
    trControl = control
    #tuneGrid = tuneGrid,
    #ntree = 500
)
predicciones_rf <- predict(modelo_rf, test_x)
matrix_confusion_rf <- confusionMatrix(predicciones_rf, test_y_factor)
f1_score_rf <- calcular_f1_score(matrix_confusion_rf)
print(f1_score_rf)
# 0.9777778

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
# COMO METRICA DE RENDIMIENTO ALTERNATIVA USAREMOS EL R2 ajustado
ind <- splitTools::partition(scaled_data$ActividadesInvoluntarias, p = c(0.5, 0.2, 0.3))
data_train <- scaled_data[ind$`1`, ]
data_valid <- scaled_data[ind$`2`, ]
data_test <- scaled_data[ind$`3`, ]

train_x <- data_train[, setdiff(seq_len(ncol(data_train)), 6)]
train_y <- data_train[, 6]
valid_x <- data_valid[, setdiff(seq_len(ncol(data_train)), 6)]
valid_y <- data_valid[, 6]
test_x <- data_test[, setdiff(seq_len(ncol(data_train)), 6)]
test_y <- data_test[, 6]

modelo_rf <- randomForest(ActividadesInvoluntarias ~ ., data = rbind(data_train, data_valid), ntree = 500, mtry = 3, importance = TRUE)
predictions <- predict(modelo_rf, test_x)

r2_rf <- 1 - sum((predictions - test_y)^2) / sum((test_y - mean(test_y))^2)
print(paste("R^2:", r2_rf))
# R^2: 0.825868921770942

# Random forest k fold 10
ind <- splitTools::partition(scaled_data$ActividadesInvoluntarias, p = c(0.7, 0.3))
data_train <- scaled_data[ind$`1`, ]
data_test <- scaled_data[ind$`2`, ]

train_x <- data_train[, setdiff(seq_len(ncol(data_train)), 6)]
train_y <- data_train[, 6]

test_x <- data_test[, setdiff(seq_len(ncol(data_train)), 6)]
test_y <- data_test[, 6]
control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

modelo_rf <- train(
    x = train_x, y = train_y,
    method = "rf",
    trControl = control
)
modelo_rf
predicciones_rf <- predict(modelo_rf, test_x)

r2_rf <- 1 - sum((predicciones_rf - test_y)^2) / sum((test_y - mean(test_y))^2)
print(paste("R^2:", r2_rf))
# 0.856340743506255
# Usando el R2 vemos que el modelo random forest para un k-fold 10 explica el 85.63% de la variabilidad de los datos

# 2.
# COMO METRICA DE RENDIMIENTO ALTERNATIVA USAREMOS EL AUC DE LA CURVA ROC
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
    x = rbind(train_x,valid_x), y = c(train_y_factor,valid_y_factor),
    method = "knn"
)
prediccion_knn <- predict(modelo_knn, newdata = test_x)
roc_auc <- calcular_auc(test_y_factor, prediccion_knn)
print(roc_auc)
# 0.9824

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
    # tuneGrid = tuneGrid,
    # ntree = 500
)
modelo_rf
predicciones_rf <- predict(modelo_rf, test_x)
roc_auc <- calcular_auc(test_y_factor, predicciones_rf)
print(roc_auc)
# 0.9577


# PARA EL MODELO KNN EN VALIDACION CRUZADA SIMPLE OBTUVIMOS UN AUC DE 0.9887
# PARA EL MODELO RANDOM FOREST EN VALIDACION CRUZADA K FOLD 10 OBTUVIMOS UN AUC DE 0.9647
# POR LO QUE PODEMOS CONCLUIR QUE EL MODELO KNN ES EL QUE MEJOR SE AJUSTA A LOS DATOS SEGUN ESTA NUEVA METRICA