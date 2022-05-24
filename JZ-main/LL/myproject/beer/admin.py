from django.contrib import admin
from .models import *
# from .models import User

# Register your models here.


class HotelAdmin(admin.ModelAdmin):
    list_display = ('id', 'place', 'name', 'rating', 'distance', 'cost',
                    'address', 'explain', 'kind', 'clean', 'conv', 'url',
                    'img', 'classfication')


class RestaurantAdmin(admin.ModelAdmin):
    list_display = ('index', 'place', 'name', 'rating', 'review',
                    'classfication', 'address', 'explain', 'url', 'img')


class TourAdmin(admin.ModelAdmin):
    list_display = ('index', 'place', 'rating', 'review', 'classfications',
                    'address', 'explain', 'mood', 'topic', 'reason', 'cluster')


class MergeAdmin(admin.ModelAdmin):
    list_display = ('index', '장소', '아이디', '평점', '평균평점', '리뷰개수', '구분', '주소',
                    '설명', 'like')


class HotelCartAdmin(admin.ModelAdmin):
    list_display = ('hotel_id', 'user', 'hotel')


class Ver3DetailAdmin(admin.ModelAdmin):
    list_display = ('id', 'together', 'theme', 'active', 'walking', 'view')


class SurveyAdmin(admin.ModelAdmin):
    list_display = ('locate', 'nature', 'food', 'reports', 'history',
                    'themepark', 'heeling', 'arts', 'city', 'family')


class Review1Admin(admin.ModelAdmin):
    list_display = ('locate', 'review_star', 'review_body')


class DetailAdmin(admin.ModelAdmin):
    list_display = ('id', 'together', 'style', 'active', 'theme', 'view')


admin.site.register(Review1, Review1Admin)

admin.site.register(Detail, DetailAdmin)

admin.site.register(Survey, SurveyAdmin)

admin.site.register(Hotel, HotelAdmin)

admin.site.register(Restaurant, RestaurantAdmin)

admin.site.register(Tour, TourAdmin)

admin.site.register(Merge, MergeAdmin)

admin.site.register(HotelCart, HotelCartAdmin)