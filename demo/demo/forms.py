from django import forms

class Economynews(forms.Form):
    news_query = forms.CharField(
        label='股票名稱或代碼', 
        max_length=100, 
        required=True,
        widget=forms.TextInput(attrs={'placeholder': ' ', 'class': 'form-control'})
    )
    news_days = forms.IntegerField(
        label='顯示的新聞數量', 
        min_value=1, 
        max_value=1000, 
        required=True,
        widget=forms.NumberInput(attrs={'placeholder': ' ', 'class': 'form-control'})
    )
    
class StocknumberInput(forms.Form):
    stock_number = forms.CharField(label='股票代碼', max_length=20)
    days = forms.IntegerField(label='資料天數', min_value=1)
    