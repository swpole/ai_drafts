import json
import argparse
import os
import re
from typing import Dict, Any, List, Set
from pathlib import Path

# Импортируем наш класс анализатора
from scan_custom_nodes import NodeMappingAnalyzer, NodeMappingResult

class ComfyUIJsonToPythonConverter:
    def __init__(self):
        self.node_templates = {}
        self.imports = set()
        self.generated_code = []
        self.node_variables = {}
        self.pending_nodes = {}  # Для узлов, которые не могут быть обработаны сразу
        self.mapping_analyzer = NodeMappingAnalyzer()
        self.node_mappings = {}  # Кэш для mapping информации

    def load_workflow_json(self, json_path: str) -> Dict[str, Any]:
        """Загружает JSON файл workflow"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def sanitize_variable_name(self, name: str) -> str:
        """Создает допустимое имя переменной из имени узла"""
        # Удаляем все не-алфавитные символы и заменяем на _
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Удаляем повторяющиеся _
        sanitized = re.sub(r'_+', '_', sanitized)
        # Удаляем _ в начале и конце
        sanitized = sanitized.strip('_')
        # Если начинается с цифры, добавляем префикс
        if sanitized and sanitized[0].isdigit():
            sanitized = 'node_' + sanitized
        return sanitized.lower()

    def get_node_dependencies(self, node_data: Dict[str, Any]) -> Set[str]:
        """Возвращает набор ID узлов, от которых зависит текущий узел"""
        dependencies = set()
        inputs = node_data.get('inputs', {})
        
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                # Это ссылка на другой узел [node_id, output_index]
                ref_node_id, output_index = input_value
                dependencies.add(ref_node_id)
        
        return dependencies

    def get_node_mapping_info(self, class_type: str) -> NodeMappingResult:
        """Получает информацию о mapping для указанного класса"""
        if not self.node_mappings:
            # Загружаем mapping информацию при первом вызове
            print("Analyzing node mappings...")
            mapping_results = self.mapping_analyzer.run_analysis()
            self.node_mappings = {result.node_name: result for result in mapping_results}
            print(f"Found {len(self.node_mappings)} node mappings")
        
        return self.node_mappings.get(class_type)

    def process_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """Обрабатывает отдельный узел и генерирует для него код.
        Возвращает True если узел успешно обработан, False если есть зависимости."""
        class_type = node_data.get('class_type', 'UnknownNode')
        inputs = node_data.get('inputs', {})
        
        # Проверяем зависимости
        dependencies = self.get_node_dependencies(node_data)
        missing_dependencies = [dep for dep in dependencies if dep not in self.node_variables]
        
        if missing_dependencies:
            print(f"Node {node_id} ({class_type}) has unmet dependencies: {missing_dependencies}")
            return False
        
        # Получаем информацию о mapping для этого типа узла
        mapping_info = self.get_node_mapping_info(class_type)
        
        if mapping_info:
            # Добавляем правильный импорт
            self.imports.add(mapping_info.import_from)
            class_name = mapping_info.class_name
            function_name = mapping_info.function_value
        else:
            # Если mapping не найден, используем стандартный подход
            print(f"Warning: No mapping found for node type: {class_type}")
            self.imports.add(f"# Unknown import for {class_type}")
            class_name = class_type
            function_name = "execute"
        
        # Создаем имя переменной для узла
        var_name = self.sanitize_variable_name(f"{class_name}_{node_id}")
        self.node_variables[node_id] = var_name
        
        # Генерируем код для создания экземпляра узла
        self.generated_code.append(f"{var_name} = {class_name}()")
        
        # Подготавливаем входные параметры
        input_params = []
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                # Это ссылка на другой узел [node_id, output_index]
                ref_node_id, output_index = input_value
                if ref_node_id in self.node_variables:
                    ref_var = self.node_variables[ref_node_id]
                    # Добавляем .detach() для тензоров, если это необходимо
                    tensor_output = f"{ref_var}_output[{output_index}]"
                    # Для некоторых узлов (например, SaveImage) нужно добавлять .detach()
                    if class_type in ["SaveImage", "PreviewImage"] and input_name == "images":
                        tensor_output += ".detach()"
                    input_params.append(f"{input_name}={tensor_output}")
                else:
                    # Эта ситуация не должна возникать после проверки зависимостей
                    print(f"Warning: Unexpected missing reference to node {ref_node_id}")
                    input_params.append(f"{input_name}=None  # Reference to unknown node {ref_node_id}")
            else:
                # Простое значение
                if isinstance(input_value, str):
                    # Экранируем кавычки в строках
                    escaped_value = input_value.replace('"', '\\"')
                    input_value = f'"{escaped_value}"'
                elif input_value is None:
                    input_value = "None"
                input_params.append(f"{input_name}={input_value}")
        
        # Генерируем вызов метода узла с правильным именем функции
        if input_params:
            params_str = ", ".join(input_params)
            self.generated_code.append(f"{var_name}_output = {var_name}.{function_name}({params_str})")
        else:
            self.generated_code.append(f"{var_name}_output = {var_name}.{function_name}()")
        
        # Добавляем сохранение выходных данных (если нужно)
        self.generated_code.append(f"# Output available as {var_name}_output")
        
        self.generated_code.append("")  # Пустая строка для читаемости
        return True

    def process_all_nodes(self, workflow_data: Dict[str, Any]):
        """Обрабатывает все узлы в правильной последовательности"""
        processed_nodes = set()
        remaining_nodes = {node_id: node_data for node_id, node_data in workflow_data.items() 
                          if not node_id.startswith('_')}
        
        max_iterations = len(remaining_nodes) * 2  # Защита от бесконечного цикла
        iteration = 0
        
        while remaining_nodes and iteration < max_iterations:
            iteration += 1
            nodes_to_process = list(remaining_nodes.items())
            
            for node_id, node_data in nodes_to_process:
                if self.process_node(node_id, node_data):
                    # Успешно обработан - удаляем из оставшихся
                    del remaining_nodes[node_id]
                    processed_nodes.add(node_id)
                    print(f"Processed node: {node_id}")
            
            # Если остались узлы, выводим информацию
            if remaining_nodes:
                print(f"Iteration {iteration}: {len(remaining_nodes)} nodes remaining")
                for node_id in list(remaining_nodes.keys()):
                    node_data = remaining_nodes[node_id]
                    dependencies = self.get_node_dependencies(node_data)
                    missing_deps = [dep for dep in dependencies if dep not in self.node_variables]
                    if missing_deps:
                        print(f"  Node {node_id} waiting for: {missing_deps}")
        
        # Обрабатываем оставшиеся узлы (если есть циклические dependencies)
        if remaining_nodes:
            print(f"Warning: {len(remaining_nodes)} nodes could not be processed due to unmet dependencies:")
            for node_id, node_data in remaining_nodes.items():
                dependencies = self.get_node_dependencies(node_data)
                missing_deps = [dep for dep in dependencies if dep not in self.node_variables]
                print(f"  Node {node_id}: missing {missing_deps}")
                
                # Все равно пытаемся обработать с заглушками
                class_type = node_data.get('class_type', 'UnknownNode')
                mapping_info = self.get_node_mapping_info(class_type)
                
                if mapping_info:
                    class_name = mapping_info.class_name
                    function_name = mapping_info.function_value
                    self.imports.add(mapping_info.import_from)
                else:
                    class_name = class_type
                    function_name = "execute"
                    self.imports.add(f"# Unknown import for {class_type}")
                
                var_name = self.sanitize_variable_name(f"{class_name}_{node_id}")
                self.node_variables[node_id] = var_name
                
                self.generated_code.append(f"# WARNING: Node {node_id} ({class_type}) has unmet dependencies")
                self.generated_code.append(f"{var_name} = {class_name}()")
                
                inputs = node_data.get('inputs', {})
                input_params = []
                for input_name, input_value in inputs.items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        ref_node_id, output_index = input_value
                        if ref_node_id in self.node_variables:
                            ref_var = self.node_variables[ref_node_id]
                            # Добавляем .detach() для тензоров в SaveImage
                            tensor_output = f"{ref_var}_output[{output_index}]"
                            if class_type in ["SaveImage", "PreviewImage"] and input_name == "images":
                                tensor_output += ".detach()"
                            input_params.append(f"{input_name}={tensor_output}")
                        else:
                            input_params.append(f"{input_name}=None  # Missing node {ref_node_id}")
                    else:
                        if isinstance(input_value, str):
                            escaped_value = input_value.replace('"', '\\"')
                            input_value = f'"{escaped_value}"'
                        input_params.append(f"{input_name}={input_value}")
                
                if input_params:
                    params_str = ", ".join(input_params)
                    self.generated_code.append(f"{var_name}_output = {var_name}.{function_name}({params_str})")
                else:
                    self.generated_code.append(f"{var_name}_output = {var_name}.{function_name}()")
                
                self.generated_code.append("")

    def generate_python_code(self, workflow_data: Dict[str, Any]) -> str:
        """Генерирует полный Python код из workflow данных"""
        self.imports.clear()
        self.generated_code.clear()
        self.node_variables.clear()
        self.pending_nodes.clear()
        
        # Предварительно загружаем mapping информацию
        self.get_node_mapping_info("dummy")  # Инициализируем кэш
        
        # Добавляем стандартные импорты
        self.imports.add("import os")
        self.imports.add("import sys")
        self.imports.add("import torch")
        
        # Добавляем комментарий в начало
        header_lines = [
            '"""',
            "Auto-generated ComfyUI workflow Python script",
            "Generated from Export (API) JSON format",
            '"""',
            ""
        ]
        
        # Обрабатываем все узлы с правильной последовательностью
        self.process_all_nodes(workflow_data)
        
        # Собираем уникальные импорты
        unique_imports = set()
        for import_stmt in self.imports:
            if import_stmt.startswith("from ") or import_stmt.startswith("import "):
                unique_imports.add(import_stmt)
            else:
                # Это комментарий или неизвестный импорт
                unique_imports.add(f"# {import_stmt}")
        
        # Добавляем базовые импорты для узлов
        unique_imports.add("import folder_paths")
        unique_imports.add("from nodes import *")
        
        # Сортируем импорты
        sorted_imports = sorted(unique_imports)
        
        # Добавляем пустую строку после импортов
        sorted_imports.append("")
        
        # Добавляем основную функцию
        main_lines = [
            "def main():",
            '    """Основная функция выполнения workflow"""',
            ""
        ]
        
        # Добавляем отступ для сгенерированного кода узлов
        node_code = ["    " + line for line in self.generated_code]
        
        # Собираем окончательный код
        final_code = header_lines + sorted_imports + main_lines + node_code
        
        # Добавляем завершение функции и вызов main
        final_code.append("")
        final_code.append("if __name__ == \"__main__\":")
        final_code.append("    main()")
        
        return "\n".join(final_code)

    def save_python_code(self, python_code: str, output_path: str):
        """Сохраняет сгенерированный Python код в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)

def main():

    input_json = "image_qwen_image.json"
    output_py = "image_qwen_image.py"
    
    # Создаем конвертер и обрабатываем файл
    converter = ComfyUIJsonToPythonConverter()
    
    try:
        # Загружаем JSON
        workflow_data = converter.load_workflow_json(input_json)

        # Генерируем Python код
        python_code = converter.generate_python_code(workflow_data)
        
        # Сохраняем результат
        converter.save_python_code(python_code, output_py)
        
        print(f"Successfully converted {input_json} to {output_py}")
        print(f"Generated {len(python_code.splitlines())} lines of code")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()
