from django.shortcuts import render, redirect
from beer.models import *
from django.contrib.auth.decorators import login_required


@login_required(login_url="/user/login")
def cart_add(request, id):
    cart = HotelCart(request)
    product = Hotel.objects.get(id=id)
    cart.add(product=product)
    return redirect("/")


@login_required(login_url="/user/login")
def item_clear(request, id):
    cart = HotelCart(request)
    product = Hotel.objects.get(id=id)
    cart.remove(product)
    return redirect("cart_detail")


@login_required(login_url="/user/login")
def cart_clear(request):
    cart = HotelCart(request)
    cart.clear()
    return redirect("cart_detail")


@login_required(login_url="/user/login")
def cart_detail(request):
    return render(request, 'home/index.html')