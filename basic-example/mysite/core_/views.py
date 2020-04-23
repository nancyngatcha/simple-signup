from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
import fibonacci
from mysite.core import generate
@login_required
def home(request):
    return render(request, 'home.html')


def signup(request):
    print("eeeeeeeeeeeeeeeeeeeeeeeeee")
    if request.method == 'POST':
        
        form = UserCreationForm(request.POST)
        print("zzzzzzzzzzzzzzzzzz")
        print(form)
        print("oooooooooooooooooooooooo")
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def fibocal(request):
    if request.method == 'POST':
       fb = int(request.POST['fcal'])
       cal = fb + 1
       return render(request, 'signup.html', {'cal': cal})
    else:
       return render(request, 'signup.html', {})

def ml(request):
    if request.method == 'POST':
       profil = list(request.POST['fcal'])
       profil = list(map(int, profil))
       print(profil)

       """profil = list(request.POST['fcal'])"""
       pred = generate.machine_learning(profil)
       return render(request, 'signup.html', {'cal': pred[0], 'stats' : pred[1]})
    else:
       return render(request, 'signup.html', {})