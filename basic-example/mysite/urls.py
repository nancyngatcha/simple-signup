from django.conf.urls import url, include
from django.contrib.auth import views as auth_views

from mysite.core import views as core_views


urlpatterns = [
    url(r'^$', core_views.home, name='home'),
    url(r'^login/$', auth_views.login, {'template_name': 'login.html'}, name='login'),
    url(r'^logout/$', auth_views.logout, {'next_page': 'login'}, name='logout'),
    url(r'^signup/$', core_views.signup, name='signup'),
    url(r'^architecte/$', core_views.architecte, name='architecte'),
    url(r'^logicien/$', core_views.logicien, name='logicien'),
    url(r'^commandant/$', core_views.commandant, name='commandant'),
    url(r'^classement/$', core_views.classement, name='classement'),
    
    
]
