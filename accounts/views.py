from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .forms import RegisterForm
from django.contrib import messages

# Create your views here.

def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)   # auto login after register
            return redirect("home")  # redirect to home or dashboard
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {"form": form})

def logout_view(request):
    if request.method == "POST":
        logout(request)
        return redirect("home")
    return redirect("home")

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("dashboard")   # or "home"
        else:
            messages.error(request, "Invalid username or password")

    # For GET requests (or failed POST), just render the form
    return render(request, "accounts/login.html")