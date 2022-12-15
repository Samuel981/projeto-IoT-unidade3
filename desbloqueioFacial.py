# USAGE
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/nome_do_video.mp4
# OU ... -e encodings.pickle -i videos/nome_do_video.mp4 -o output/nome_do_video.avi -d 0

import face_recognition # type: ignore
import argparse # type: ignore
import imutils # type: ignore
import time
import pickle
import os
import sys
import access
import cv2 # type: ignore
from base64 import b64encode
from datetime import datetime
from Adafruit_IO import MQTTClient # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
args = vars(ap.parse_args())

# aio = Client(username=access.USER, key=access.KEY)
grupo = 'desbloqueiofacial.'
feedTentativa = grupo+'tentativadedesbloqueio'
feedAcesso = grupo+'sistemadeacesso'
feedRegistro = grupo+'registro'
feedCaptura = grupo+'captura'

def connected(client):
    client.subscribe(feedTentativa, access.USER)
    
def disconnected(client):
    print('Disconnected from Adafruit IO!')
    sys.exit(1)
    
tentativa = False
def message(client, feed_id, payload):
    global tentativa
    tentativa = not tentativa

os.system('cls')
print("------- [ Sistema de desbloqueio facial ] --------\n")
print("[INFO] Carregando encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# Create an MQTT client instance.
print("[INFO] Conetando com o Adafruit_IO...")
client = MQTTClient(access.USER, access.KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
# Connect to the Adafruit IO server.
client.connect()
client.loop_background()
stream = cv2.VideoCapture(args["input"])
writer = None

atraso = 1
if stream is None:
    print("[INFO] Recebendo imagens ao vivo...")
    stream = cv2.VideoCapture(0)
    atraso = 50
else:
    print("[INFO] Recebendo imagens...")
    fps = round(stream.get(cv2.CAP_PROP_FPS))
    atraso = int(1000/fps) #1 segundo (1000mili) / taxa de quadros
    
nomeJanela = "Frame"
cv2.namedWindow(nomeJanela, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(nomeJanela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("[INFO] Iniciando transmissão...")

# inicia a contagem de frames
frame = 0

# loop over frames from the video file stream
while True:  
    # grab the next frame
    (grabbed, frame) = stream.read()

    # fim de transmissao
    if frame is None:
        print("[INFO] Encerrando...")
        break

    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    if(tentativa == True):
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                    encoding)
            name = "Desconhecido"

            # melhorar isso para aceitar uma pessoa por vez
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                        (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            
            if(name != "Desconhecido"):
                client.publish(feedAcesso, "ON")
                client.publish(feedRegistro, "Desbloqueado por "+name)
            else:
                client.publish(feedRegistro, "Tentativa de desbloqueio por <desconhecido>")
                dataAtual = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                cv2.imwrite("images/nao_autorizado_"+dataAtual+".jpg", frame)
                with open("images/nao_autorizado_"+dataAtual+".jpg", 'rb') as imageFile:
                    img = b64encode(imageFile.read())
                    client.publish(feedCaptura,  img.decode('utf-8'))
                imageFile.close()
            client.publish(feedTentativa, "False")
            # tentativa = False

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow(nomeJanela, frame)
        key = cv2.waitKey(atraso) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("[INFO] Encerrando...")
            break
    frame += 1

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
