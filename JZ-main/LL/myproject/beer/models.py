from django.db import models
from .models import *
from django.urls import reverse


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'), )


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'), )


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'), )


class AuthUserUserPermissions(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'), )


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType',
                                     models.DO_NOTHING,
                                     blank=True,
                                     null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'), )


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Hotel(models.Model):
    id = models.IntegerField(primary_key=True)
    place = models.CharField(max_length=45, blank=True, null=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    distance = models.FloatField(blank=True, null=True)
    cost = models.IntegerField(blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    explain = models.TextField(blank=True, null=True)
    kind = models.FloatField(blank=True, null=True)
    clean = models.FloatField(blank=True, null=True)
    conv = models.FloatField(blank=True, null=True)
    url = models.TextField(blank=True, null=True)
    img = models.TextField(blank=True, null=True)
    classfication = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'hotel'


class Tour(models.Model):
    index = models.OneToOneField('Merge',
                                 models.DO_NOTHING,
                                 db_column='index',
                                 primary_key=True)
    place = models.CharField(max_length=45, blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    review = models.TextField(blank=True, null=True)
    classfications = models.CharField(max_length=45, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    explain = models.CharField(max_length=255, blank=True, null=True)
    mood = models.CharField(max_length=255, blank=True, null=True)
    topic = models.CharField(max_length=255, blank=True, null=True)
    reason = models.CharField(max_length=255, blank=True, null=True)
    cluster = models.IntegerField(db_column='Cluster', blank=True,
                                  null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'tour'


class Restaurant(models.Model):
    index = models.IntegerField(primary_key=True)
    place = models.CharField(max_length=45, blank=True, null=True)
    name = models.CharField(max_length=45, blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    review = models.IntegerField(blank=True, null=True)
    classfication = models.CharField(max_length=255, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    explain = models.CharField(max_length=255, blank=True, null=True)
    url = models.TextField(blank=True, null=True)
    img = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'restaurant'


class Merge(models.Model):
    index = models.IntegerField(primary_key=True)
    장소 = models.CharField(max_length=255, blank=True, null=True)
    아이디 = models.CharField(max_length=255, blank=True, null=True)
    평점 = models.IntegerField(blank=True, null=True)
    평균평점 = models.FloatField(blank=True, null=True)
    리뷰개수 = models.CharField(max_length=255, blank=True, null=True)
    구분 = models.CharField(max_length=255, blank=True, null=True)
    주소 = models.CharField(max_length=255, blank=True, null=True)
    설명 = models.CharField(max_length=255, blank=True, null=True)
    like = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'merge'


class Review(models.Model):
    review_id = models.IntegerField(primary_key=True)
    tourname = models.ForeignKey('Tour',
                                 models.DO_NOTHING,
                                 db_column='tourname',
                                 blank=True,
                                 null=True,
                                 related_name='+')
    user = models.ForeignKey('user.User',
                             models.DO_NOTHING,
                             blank=True,
                             null=True)
    review_title = models.CharField(max_length=45, blank=True, null=True)
    review_body = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'review'


class Detail(models.Model):
    id = models.AutoField(primary_key=True)
    together = models.CharField(max_length=45, blank=True, null=True)
    style = models.CharField(max_length=45, blank=True, null=True)
    active = models.CharField(max_length=45, blank=True, null=True)
    theme = models.CharField(max_length=45, blank=True, null=True)
    view = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Detail'


class Survey(models.Model):
    nature = models.IntegerField(blank=True, null=True, default=0)
    food = models.IntegerField(blank=True, null=True, default=0)
    reports = models.IntegerField(blank=True, null=True, default=0)
    history = models.IntegerField(blank=True, null=True, default=0)
    themepark = models.IntegerField(blank=True, null=True, default=0)
    heeling = models.IntegerField(blank=True, null=True, default=0)
    arts = models.IntegerField(blank=True, null=True, default=0)
    city = models.IntegerField(blank=True, null=True, default=0)
    family = models.IntegerField(blank=True, null=True, default=0)
    locate = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'survey'


class Review1(models.Model):
    locate = models.CharField(max_length=100, blank=True, null=True)
    review_star = models.IntegerField(blank=True, null=True)
    review_body = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'review1'


class HotelCart(models.Model):
    hotelcart_id = models.IntegerField(primary_key=True)
    user = models.ForeignKey('user.User',
                             models.DO_NOTHING,
                             blank=True,
                             null=True)
    hotel = models.ForeignKey(Hotel, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'hotel_cart'


class Survey1(models.Model):
    locate = models.CharField(max_length=100, blank=True, null=True, default=0)
    family = models.IntegerField(blank=True, null=True, default=0)
    reports = models.IntegerField(blank=True, null=True, default=0)
    history = models.IntegerField(blank=True, null=True, default=0)
    themepark = models.IntegerField(blank=True, null=True, default=0)
    food = models.IntegerField(blank=True, null=True, default=0)
    arts = models.IntegerField(blank=True, null=True, default=0)
    heeling = models.IntegerField(blank=True, null=True, default=0)
    city = models.IntegerField(blank=True, null=True, default=0)
    nature = models.IntegerField(blank=True, null=True, default=0)

    class Meta:
        managed = False
        db_table = 'survey1'