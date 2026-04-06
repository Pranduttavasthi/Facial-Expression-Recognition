import torch
import cv2
import timm
import numpy as np
from torchvision import transforms
import time
import textwrap
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

emotion_score_map = {
    'Happy': -2,
    'Neutral': 0,
    'Surprise': -1,
    'Sad': 2,
    'Angry': 2,
    'Fear': 2,
    'Disgust': 2
}

quotes = {

"Very Relaxed":
"""Your emotional signals indicate a calm and balanced mental state.
You appear relaxed and mentally composed during the observation period.
Maintaining this level of calmness is excellent for both mental and physical health.
Continue engaging in activities that support relaxation and positivity such as
exercise, social interaction, hobbies, and mindful breathing.""" ,

"Normal":
"""Your emotional responses appear balanced with moderate fluctuations.
This is considered a healthy and normal psychological state where the mind
responds naturally to different stimuli. To maintain emotional balance,
consider short breaks, deep breathing, staying hydrated, and maintaining
positive social interactions throughout your day.""" ,

"High Stress":
"""Your emotional signals indicate elevated stress levels during the session.
High stress can impact concentration, mood, and overall well-being.
It is recommended to take a short break, practice slow breathing,
stretch your body, or step away from the screen for a few minutes.
Consistent relaxation techniques and healthy sleep can significantly help
reduce long-term stress levels."""
}

stress_score = 50
duration = None
start_time = None

session_finished = False
menu_screen = True

def create_particles(count,width,height):

    particles=[]

    for i in range(count):

        particles.append({

            "x":random.randint(0,width),
            "y":random.randint(0,height),
            "dx":random.uniform(-3,3),
            "dy":random.uniform(-3,3),
            "size":random.randint(12,20)

        })

    return particles

def draw_happy(frame,x,y,size):

    cv2.circle(frame,(x,y),size,(0,255,0),-1)
    cv2.circle(frame,(x-5,y-3),2,(0,0,0),-1)
    cv2.circle(frame,(x+5,y-3),2,(0,0,0),-1)
    cv2.ellipse(frame,(x,y+2),(6,4),0,0,180,(0,0,0),2)


def draw_neutral(frame,x,y,size):

    cv2.circle(frame,(x,y),size,(0,200,255),-1)
    cv2.circle(frame,(x-5,y-3),2,(0,0,0),-1)
    cv2.circle(frame,(x+5,y-3),2,(0,0,0),-1)
    cv2.line(frame,(x-5,y+5),(x+5,y+5),(0,0,0),2)


def draw_scary(frame,x,y,size):

    cv2.circle(frame,(x,y),size,(0,0,255),-1)
    cv2.circle(frame,(x-5,y-3),3,(0,0,0),-1)
    cv2.circle(frame,(x+5,y-3),3,(0,0,0),-1)
    cv2.ellipse(frame,(x,y+6),(6,6),0,0,360,(0,0,0),2)


def draw_particles(frame,particles,mode):

    h,w,_ = frame.shape

    for p in particles:

        p["x"] += p["dx"]
        p["y"] += p["dy"]

        if p["x"]<0 or p["x"]>w:
            p["dx"] *= -1
        if p["y"]<0 or p["y"]>h:
            p["dy"] *= -1

        if mode=="happy":
            draw_happy(frame,int(p["x"]),int(p["y"]),p["size"])

        elif mode=="neutral":
            draw_neutral(frame,int(p["x"]),int(p["y"]),p["size"])

        else:
            draw_scary(frame,int(p["x"]),int(p["y"]),p["size"])

def draw_box_text(img,text,y,box_color,scale=0.9):

    (w,h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)

    x=(img.shape[1]-w)//2

    cv2.rectangle(img,(x-15,y-h-15),(x+w+15,y+15),box_color,-1)

    cv2.putText(img,text,(x,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,(255,255,255),2)

def draw_multiline_text_center(img,text,y):

    lines=textwrap.wrap(text,width=60)

    for i,line in enumerate(lines):

        (w,h),_=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)

        x=(img.shape[1]-w)//2
        y_line=y+(i*28)

        cv2.putText(img,line,(x+2,y_line+2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,(0,0,0),4)

        cv2.putText(img,line,(x,y_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,(255,255,255),2)

class CustomSwinTransformer(torch.nn.Module):
    def __init__(self,num_classes=7):

        super().__init__()

        self.backbone=timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=0
        )

        self.classifier=torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(512,num_classes)
        )

    def forward(self,x):

        x=self.backbone(x)

        return self.classifier(x)

model=CustomSwinTransformer()
model.load_state_dict(torch.load("best_model (4).pth",map_location=device))
model.to(device)
model.eval()

transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

cap=cv2.VideoCapture(0)

particles=create_particles(100,1280,720)

while True:

    ret,frame=cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    if menu_screen:
        blurred=cv2.GaussianBlur(frame,(35,35),0)
        draw_box_text(blurred,"Mental Health Stress Test",100,(120,0,120),1.0)
        draw_box_text(blurred,"Select Timer",170,(0,100,150))

        draw_box_text(blurred,"Press 1 : 20 Seconds",230,(60,60,60))
        draw_box_text(blurred,"Press 2 : 30 Seconds",270,(60,60,60))
        draw_box_text(blurred,"Press 3 : 45 Seconds",310,(60,60,60))
        draw_box_text(blurred,"Press 4 : 60 Seconds",350,(60,60,60))
        frame=blurred
    elif not session_finished:

        if start_time is None:
            start_time=time.time()

        elapsed_time=int(time.time()-start_time)
        remaining_time=max(0,duration-elapsed_time)

        if elapsed_time>=duration:
            session_finished=True
        for (x,y,w,h) in faces:

            face=frame[y:y+h,x:x+w]
            rgb=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

            input_tensor=transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                output=model(input_tensor)
                _,pred=torch.max(output,1)

                emotion=class_names[pred.item()]

            stress_score+=emotion_score_map[emotion]
            stress_score=max(0,min(100,stress_score))

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

            cv2.putText(frame,f"{emotion}",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(255,255,0),2)

        cv2.rectangle(frame,(10,10),(350,90),(0,0,0),-1)

        cv2.putText(frame,f"Time Remaining: {remaining_time}s",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)

        cv2.putText(frame,f"Stress Score: {stress_score}",
                    (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)


    if session_finished:
        blurred=cv2.GaussianBlur(frame,(35,35),0)

        if stress_score<=30:
            status="Very Relaxed"
            mode="happy"
            color=(0,200,0)

        elif stress_score<=60:
            status="Normal"
            mode="neutral"
            color=(0,200,255)

        else:
            status="High Stress"
            mode="scary"
            color=(0,0,255)

            red_overlay=np.full(frame.shape,(0,0,120),dtype=np.uint8)
            blurred=cv2.addWeighted(blurred,0.7,red_overlay,0.3,0)

        draw_particles(blurred,particles,mode)

        draw_box_text(blurred,"Session Finished",90,(120,0,120))
        draw_box_text(blurred,f"Stress Score: {stress_score}",140,(0,100,150))
        draw_box_text(blurred,f"Status: {status}",190,color)

        draw_multiline_text_center(blurred,quotes[status],250)

        draw_box_text(blurred,"Press R to Restart",520,(120,120,0))

        frame=blurred
    cv2.imshow("Mental Health Stress Detection",frame)

    key=cv2.waitKey(1)&0xFF

    if key==ord('q'):
        break

    if menu_screen:

        if key==ord('1'):
            duration=20
        elif key==ord('2'):
            duration=30
        elif key==ord('3'):
            duration=45
        elif key==ord('4'):
            duration=60
        else:
            duration=None

        if duration:
            menu_screen=False
            start_time=None
            stress_score=50
            session_finished=False

    if key==ord('r') and session_finished:

        menu_screen=True
        start_time=None
        stress_score=50
        session_finished=False
        duration=None

cap.release()
cv2.destroyAllWindows()