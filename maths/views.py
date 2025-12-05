# maths/views.py
from django.shortcuts import render

def classpad_main(request):
    """
    UI-only mock of the Casio ClassPad 'Main' screen.
    (No real CAS yet – the EXE button just echoes the input as a fake result.)
    """
    return render(request, "maths/classpad_main.html")
