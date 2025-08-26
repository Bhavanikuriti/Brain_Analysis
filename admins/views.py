from django.shortcuts import render,redirect
from django.contrib import messages
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def AdminHome(request):
    return render(request, 'admins/AdminHome.html',{})

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        messages.success(request, 'User Activated successfully.') 
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})


from django.shortcuts import redirect, render
from django.contrib import messages


def deleteuser(request):
    if request.method == 'GET':
        # Retrieve the 'uid' parameter from the request
        user_id = request.GET.get('uid')
        
        # Check if the user ID is provided and exists in the database
        if user_id:
            try:
                user = UserRegistrationModel.objects.get(id=user_id)  # Retrieve the user
                user.delete()  # Delete the user
                messages.success(request, 'User deleted successfully.')  # Success message
            except UserRegistrationModel.DoesNotExist:
                messages.error(request, 'User not found.')  # Error message if user does not exist
        else:
            messages.error(request, 'No user ID provided.')  # Error message if no ID is provided

        # Redirect to the ActivaUsers view after the operation
        return redirect(ActivaUsers)  


from django.shortcuts import render
from users.forms import UserRegistrationForm


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})