import cv2
from keras.models import load_model
import numpy as np

t = 150;

modelo = load_model('modelo_tumores.keras')
# radiografia ='tumores/appcnn/val/no/no1350.jpg'
radiografia ='tumores/appcnn/val/yes/y1371.jpg'


#Prediccion
I = cv2.imread(radiografia)
I_resz = cv2.resize(I,dsize=(t,t), interpolation=cv2.INTER_CUBIC)
I_resz = I_resz.astype('float')

if round(modelo.predict(np.array([I_resz]))[0][0])==1:
    print("Positivo")
    cv2.putText(I, 'Positivo', (0,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)
else:
    print("Negativo")
    cv2.putText(I, 'Negativo', (0,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)

cv2.imshow('Prediccion', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
