FROM nginx:1.27-alpine

# 移除預設的 default.conf
RUN rm /etc/nginx/conf.d/default.conf

# 複製反向代理與安全配置
COPY nginx.conf /etc/nginx/conf.d/
