import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import ssl
from pathlib import Path
import datetime

class NginxManagerPro:
    def __init__(self, base_path=None):
        """Инициализация менеджера Nginx"""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.nginx_dir = self.base_path / "nginx"
        self.nginx_exe = self.nginx_dir / "nginx.exe"
        self.conf_dir = self.nginx_dir / "conf"
        self.ssl_dir = self.conf_dir / "ssl"
        
    def download_nginx(self):
        """Скачивание и распаковка Nginx"""
        """Проверка, установлен ли уже Nginx"""
        if self.nginx_dir.exists() and self.nginx_exe.exists():
            print("Nginx уже установлен, пропускаем скачивание")
            return True
        
        nginx_url = "https://nginx.org/download/nginx-1.29.3.zip"
        zip_path =  Path(__file__).parent / "nginx.zip"
        
        try:
            
            # Создаем контекст без проверки SSL (на случай проблем с сертификатами)
            ssl_context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(nginx_url, context=ssl_context) as response:
                with open(zip_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Получаем имя корневой папки в архиве
                root_dir = Path(zip_ref.namelist()[0]).parts[0]
                zip_ref.extractall(self.base_path)
            
            # Переименовываем распакованную папку в просто "nginx"
            extracted_dir = self.base_path / root_dir
            if extracted_dir.exists():
                if self.nginx_dir.exists():
                    shutil.rmtree(self.nginx_dir)
                extracted_dir.rename(self.nginx_dir)
            
            # Удаляем временный файл
            zip_path.unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            return False
    
    def create_directories(self):
        """Создание необходимых директорий"""
        directories = [self.ssl_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        return True
    
    def create_ssl_certificates(self):
        """Создание самоподписанных SSL сертификатов"""
        try:
            
            # Команды для создания сертификатов через OpenSSL
            key_path = self.ssl_dir / "nginx.key"
            crt_path = self.ssl_dir / "nginx.crt"
            
            # Альтернативный способ - используем встроенный модуль ssl
            try:
                from cryptography import x509
                from cryptography.x509.oid import NameOID
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                
                # Генерируем приватный ключ
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                )
                
                # Создаем самоподписанный сертификат
                subject = issuer = x509.Name([
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "NY"),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, "NYC"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Development"),
                    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
                ])
                
                certificate = x509.CertificateBuilder().subject_name(
                    subject
                ).issuer_name(
                    issuer
                ).public_key(
                    private_key.public_key()
                ).serial_number(
                    x509.random_serial_number()
                ).not_valid_before(
                    datetime.datetime.utcnow()
                ).not_valid_after(
                    datetime.datetime.utcnow() + datetime.timedelta(days=365)
                ).add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                    ]),
                    critical=False,
                ).sign(private_key, hashes.SHA256())
                
                # Сохраняем приватный ключ
                with open(key_path, "wb") as f:
                    f.write(private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    ))
                
                # Сохраняем сертификат
                with open(crt_path, "wb") as f:
                    f.write(certificate.public_bytes(serialization.Encoding.PEM))
                
                
            except ImportError:
                # Создаем пустые файлы как заглушки
                key_path.write_text("")
                crt_path.write_text("")
                
            return True
            
        except Exception as e:
            return False
    
    def create_nginx_config(self, gradio_port=7860):
        """Создание конфигурации Nginx для Gradio"""
        config_content = f'''worker_processes  1;

events {{
    worker_connections  1024;
}}

http {{
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
    
    # Настройки для прокси
    proxy_connect_timeout 7d;
    proxy_send_timeout 7d;
    proxy_read_timeout 7d;
    
    # Для WebSocket соединений Gradio
    map $http_upgrade $connection_upgrade {{
        default upgrade;
        '' close;
    }}

    # HTTP сервер - редирект на HTTPS
    server {{
        listen       80;
        server_name  _;
        
        # Редирект всех HTTP запросов на HTTPS
        return 301 https://$host$request_uri;
    }}

    # HTTPS сервер
    server {{
        listen       443 ssl;
        server_name  _;

        # SSL сертификаты
        ssl_certificate      ssl/nginx.crt;
        ssl_certificate_key  ssl/nginx.key;

        # Настройки SSL
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
        ssl_prefer_server_ciphers off;

        # Проксирование на Gradio сервер
        location / {{
            proxy_pass http://127.0.0.1:{gradio_port};
            
            # Базовые заголовки
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Forwarded-Host $host;
            
            # WebSocket поддержка
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            
            # Отключение буферизации для SSE
            proxy_buffering off;
            
            # Разрешаем большие файлы
            client_max_body_size 100M;
        }}
    }}
}}
'''
        
        config_path = self.conf_dir / "nginx.conf"
        config_path.write_text(config_content, encoding='utf-8')
        
        return True
    
    def start_nginx(self):
        """Запуск Nginx"""
        try:
            # Останавливаем предыдущие процессы
            self.stop_nginx()
            
            # Проверяем конфигурацию
            check_result = subprocess.run(
                [str(self.nginx_exe), "-t"], 
                cwd=self.nginx_dir,
                capture_output=True, 
                text=True
            )
            
            if check_result.returncode != 0:
                return False
            
            # Запускаем Nginx
            subprocess.Popen(
                [str(self.nginx_exe)], 
                cwd=self.nginx_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            return True
            
        except Exception as e:
            return False
    
    def stop_nginx(self):
        """Остановка Nginx"""
        try:
            # Плавная остановка
            subprocess.run(
                [str(self.nginx_exe), "-s", "quit"], 
                cwd=self.nginx_dir,
                capture_output=True,
                timeout=5
            )
        except:
            pass
        
        # Принудительная остановка
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", "nginx.exe"],
                capture_output=True,
                timeout=5
            )
        except:
            pass
    
    def install_and_start(self, gradio_port=7860):
        """Полная установка и запуск"""
        
        steps = [
            ("Скачивание Nginx", self.download_nginx),            
            ("Создание директорий", self.create_directories),
            ("Создание SSL сертификатов", self.create_ssl_certificates),
            ("Создание конфигурации", lambda: self.create_nginx_config(gradio_port)),
            ("Запуск Nginx", self.start_nginx)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                return False
        return True
    
    def get_status(self):
        """Проверка статуса Nginx"""
        try:
            result = subprocess.run(
                ["tasklist", "/fi", "imagename eq nginx.exe"],
                capture_output=True,
                text=True
            )
            return "nginx.exe" in result.stdout
        except:
            return False
        
if __name__ == "__main__":
    # Создаем менеджер
    nginx_manager = NginxManagerPro()
    
    # Полная установка и запуск
    success = nginx_manager.install_and_start(gradio_port=7860)
    
    if success:
        print("✅ Nginx успешно установлен и запущен!")
        print("📊 Статус:", "Запущен" if nginx_manager.get_status() else "Остановлен")
        print("🌐 Ваш Gradio будет доступен по:")
        print("   HTTP: http://ваш-ip (редирект на HTTPS)")
        print("   HTTPS: https://ваш-ip")
        print("   Сервер остановлен, так как программа завершила работу.")
    else:
        print("❌ Ошибка при установке Nginx")