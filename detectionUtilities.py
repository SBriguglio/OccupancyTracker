from tkinter import *
import os

#Occupancy Counter
def Program1():
    #Navidate to the directory
    os.chdir("HumanOccupancyCounter")
    #execute via shell
    os.system('python3 occupancyDetection.py')
    #Naviagte back to the root
    os.chdir("..")

#Register dictation
def Program2():
    os.system('python button.py')#change name for the program

#Face Detection
def Program3():
    #Navidate to the directory
    os.chdir("FaceDetection")
    #execute via shell
    os.system('python3 faceDetection.py')
    #Naviagte back to the root
    os.chdir("..")

window=Tk()

btn=Button(window, text="LAUNCH TRACKER", fg='black' , command= Program1)
btn.place(x=70, y=100,  height=28, width=300)

btn=Button(window, text="LAUNCH REGISTER DICTATION", fg='black', command= Program2)
btn.place(x=70, y=150,  height=28, width=300)

btn=Button(window, text="FACE DETECTOR", fg='black', command= Program3)
btn.place(x=70, y=200,  height=28, width=300)

lbl=Label(window, text="HUMAN OCCUPANCY TRACKER", fg='Black',bg="SteelBlue1", font=("Impact", 30, "bold"))
lbl.place(x=45, y=30)

window.title('Detection Utilities')
window.geometry("450x270+10+10")
window.configure(bg = 'SteelBlue1')
window.mainloop()
