# hand_tracking_samples_D400-master-Miguel

Sin camara se pueden ejecutar:

* [train-hand-pose-cnn]() - entrena una CNN con los datasets seleccionados, actualmente estan seleccionados para ello algunos del gesto de puño abierto y palma con dedos juntos. 
* [synthetic-hand-tracker](./synthetic-hand-tracker/) - usando la CNN se simula la lectura del dataset que queramos. 

La CNN que se use en cada momento se puede seleccionar en el archivo "handtrack.m" encontrado en la carpeta include.
