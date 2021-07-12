# hand_tracking_samples_D400-master-Miguel

Sin camara se pueden ejecutar:

* [train-hand-pose-cnn]() - entrena una CNN con los datasets seleccionados, actualmente estan seleccionados para ello algunos del gesto de puño abierto y palma con dedos juntos. 
* [synthetic-hand-tracker](./synthetic-hand-tracker/) - usando la CNN se simula la lectura del dataset que queramos. 

La CNN que se use en cada momento se puede seleccionar en el archivo "handtrack.m" encontrado en la carpeta [include]().

La carpeta de  [datasets]() contiene todas las sesiones grabadas hasta el momento. Clasificadas en subcarpetas cada una correspondiente a un gesto. 

A su vez, la carpeta de [assets]() contiene todos los archivos ".cnnb" correspondientes a las CNNs entrenadas hasta el momento.
