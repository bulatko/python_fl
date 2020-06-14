from tkinter import *
from tkinter import scrolledtext, messagebox
import requests
import time
import asyncio
from threading import Thread    
import webbrowser

    
stop_flag = 0
    
def alert(text):
    log.insert(END, text)
    
def start():  
    if txt_url.get() == '':
        messagebox.showinfo("Ошибка", "Введите URL")
        return
    
    my_thread = MyThread('start')
    my_thread.start()
    
    
def stop():  
    my_thread = MyThread('stop')
    my_thread.start()  

    
    
class MyThread(Thread):
    
    def __init__(self, name):
        
        Thread.__init__(self)
        self.name = name
    
    def run(self):
        global stop_flag
        if(self.name == 'start'):
            stop_flag = 0
            url = txt_url.get()
            alert("Старт\n")
            interval = int(txt_interval.get())
            alert(("Адрес сайта: {}\n" + \
                 "Интервал: {} сек.\n\n").format(url, interval))
            i = 1
            while not stop_flag:
                try:
                    res = requests.get(url)
                    alert("{})Код ответа: {}\n".format(i, res.status_code))
                    time.sleep(interval)
                    i += 1
                except Exception as e:
                    alert("Ошибка: {}\nЦикл закончен".format(e))
                    stop_flag = 1
                    return
        else:
            if not stop_flag:
                stop_flag = 1
                alert('Цикл остановлен')
          
 
  


window = Tk()  
window.title("Program")  
window.geometry('500x450')  
lbl_url = Label(window, text="URL")  
lbl_url.grid(column=0, row=0, sticky=W)
txt_url = Entry(window, width=60)
txt_url.grid(padx=0, column=1, row=0, columnspan=3, sticky=W)

lbl_interval = Label(window, text="Интервал")  
lbl_interval.grid(padx='0', column=0, row=1, sticky=W)  
txt_interval = Spinbox(window, width=7, from_=1, to=100000)  
txt_interval.grid(padx='0', column=1, row=1, sticky=W)    

log = scrolledtext.ScrolledText(window, width=50, height=20)
log.grid(column=0, row=4, columnspan=4, sticky=W)  
btn_start = Button(window, text="Старт", command=start)  
btn_start.grid(column=0, row=2, sticky=W)  
btn_stop = Button(window, text="Стоп", command=stop)  
btn_stop.grid(column=1, row=2, sticky=W)  
window.mainloop()