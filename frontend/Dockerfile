FROM node as builder
WORKDIR /usr/share/app/frontend
COPY . .
RUN npm i
RUN npm run build

FROM nginx:1.16.0-alpine
RUN rm /etc/nginx/conf.d/default.conf 
COPY ./default.conf /etc/nginx/conf.d/nginx.conf
COPY --from=builder /usr/share/app/frontend/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

