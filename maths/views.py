from django.shortcuts import render

# Create your views here.
def maths(request):
    return render(request, 'maths/maths.html')

