from django.contrib import admin
from django.contrib.admin import AdminSite
from .models import UserProfile, ModelPerformance, MelasmaReport

# Customize admin site
admin.site.site_header = "MelaScan Administration"
admin.site.site_title = "MelaScan Admin"
admin.site.index_title = "Welcome to MelaScan Administration"


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'gender', 'date_of_birth']
    search_fields = ['name', 'user__username', 'user__email']


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'is_active']
    list_filter = ['is_active']
    search_fields = ['model_name']
    ordering = ['-accuracy']


@admin.register(MelasmaReport)
class MelasmaReportAdmin(admin.ModelAdmin):
    list_display = ['user', 'result', 'model_used', 'accuracy', 'date']
    list_filter = ['result', 'model_used', 'date']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['date']
    ordering = ['-date']

