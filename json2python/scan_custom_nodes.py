import sys
import ast
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple, NamedTuple


class NodeMappingResult(NamedTuple):
    """Результат анализа нода"""
    node_name: str
    class_name: str
    function_value: str
    file_path: Path
    import_path: str
    import_from: str


class NodeMappingAnalyzer:
    """
    Класс для анализа Node Class Mappings в ComfyUI кастомных нодах
    """
    
    def __init__(self):
        self.processed_files = set()
    
    def find_comfyui(self) -> Path:
        """
        Находит путь к папке ComfyUI
        """
        python_executable = sys.executable
        python_path = Path(python_executable)
        parent_dir = python_path.parent.parent
        comfyui_path = parent_dir / "ComfyUI"
        
        if not comfyui_path.exists():
            raise FileNotFoundError(f"Папка ComfyUI не найдена по пути: {comfyui_path}")
        
        if not comfyui_path.is_dir():
            raise FileNotFoundError(f"Путь существует, но это не папка: {comfyui_path}")
        
        return comfyui_path

    def parse_extra_model_paths(self, file_path: Path, filename: str = "extra_model_paths.yaml") -> Optional[List[str]]:
        """
        Парсит YAML файл с дополнительными путями к кастомным нодам
        """
        full_path = Path(file_path) / filename
        
        if not full_path.exists():
            return None
        
        if not full_path.is_file():
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            return None
        
        # Извлекаем значения comfyui:custom_nodes
        if data and 'comfyui' in data and 'custom_nodes' in data['comfyui']:
            out = data['comfyui']['custom_nodes']
            return out.splitlines()
        
        if data and 'other_ui' in data and 'custom_nodes' in data['other_ui']:
            out = data['other_ui']['custom_nodes']
            return out.splitlines()
        
        return None

    def get_all_subdirectories(self, path_list: List[Path]) -> List[Path]:
        """
        Получает все поддиректории из списка путей
        """
        result = []    
        
        for path in path_list:
            if path.exists() and path.is_dir():
                for item in path.iterdir():
                    if item.is_dir():
                        result.append(item)
        
        return result

    def find_node_class_mappings_with_imports(self, directories: List[Path]) -> List[NodeMappingResult]:
        """
        Проходит по всем папкам из списка, ищет файлы .py, извлекает информацию о NODE_CLASS_MAPPINGS
        включая импорты, и находит значение FUNCTION в соответствующих классах.
        """
        results = []
        self.processed_files.clear()
        
        for directory in directories:
            if not directory.exists() or not directory.is_dir():
                continue
                
            print(f"Анализируем директорию: {directory}")
            
            # Добавляем путь к родительской директории для импорта
            if str(directory.parent) not in sys.path:
                sys.path.insert(0, str(directory.parent))
            
            # Ищем все Python файлы в директории
            py_files = list(directory.glob("*.py"))
            
            for py_file in py_files:
                if py_file in self.processed_files:
                    continue
                    
                file_results = self._analyze_py_file(py_file, directory)
                results.extend(file_results)
        
        return results

    def _analyze_py_file(self, py_file: Path, base_dir: Path) -> List[NodeMappingResult]:
        """
        Анализирует Python файл и возвращает найденные mapping
        """
        results = []
        
        try:
            print(f"  Анализируем файл: {py_file.name}")
            
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self.processed_files.add(py_file)
            
            # Ищем NODE_CLASS_MAPPINGS в текущем файле
            local_mappings = self._find_node_class_mappings_in_ast(tree)
            
            # Ищем импорты для дальнейшего анализа
            imports = self._find_imports_with_mappings(tree)
            
            # Анализируем импортированные файлы
            for import_info in imports:
                imported_file = self._find_imported_file(import_info, base_dir, py_file)
                if imported_file and imported_file not in self.processed_files:
                    print(f"    Переходим к импортированному файлу: {imported_file.name}")
                    imported_results = self._analyze_py_file(imported_file, base_dir)
                    results.extend(imported_results)
            
            # Если нашли mapping в текущем файле, ищем FUNCTION в классах
            if local_mappings:
                print(f"    Найдены mapping в {py_file.name}: {local_mappings}")
                
                # Ищем FUNCTION атрибуты в классах
                class_functions = self._find_class_function_values(tree)
                
                # Генерируем import path для текущего файла
                import_path = self._generate_import_path(py_file, base_dir.parent)
                if py_file == self.find_comfyui() / "nodes.py":
                    import_path = "nodes"
                
                # Сопоставляем результаты
                for node_name, class_name in local_mappings.items():
                    function_name = class_functions.get(class_name)
                    if function_name:  # Если FUNCTION найден (любое значение)
                        import_from = f"from {import_path} import {class_name}"
                        result = NodeMappingResult(
                            node_name=node_name,
                            class_name=class_name,
                            function_value=function_name,
                            file_path=py_file,
                            import_path=import_path,
                            import_from=import_from
                        )
                        results.append(result)
                        print(f"      Найдено: {node_name} -> {class_name} -> FUNCTION = '{function_name}'")
                        print(f"      Путь: {import_path}")
                    else:
                        print(f"      FUNCTION не найден для класса: {class_name}")
            
            return results
            
        except Exception as e:
            print(f"    Ошибка при анализе файла {py_file}: {e}")
            return []

    def _generate_import_path(self, py_file: Path, base_dir: Path) -> str:
        """
        Генерирует import path для файла относительно базовой директории
        """
        try:
            # Получаем относительный путь от base_dir к файлу
            relative_path = py_file.relative_to(base_dir)
            
            # Преобразуем путь в формат импорта
            import_parts = []
            
            # Обрабатываем путь до файла
            for part in relative_path.parent.parts:
                if part != '__pycache__' and not part.endswith('.py'):
                    import_parts.append(part)
            
            # Добавляем имя модуля (без .py)
            module_name = py_file.stem
            if module_name != '__init__':
                import_parts.append(module_name)
            
            return '.'.join(import_parts)
            
        except ValueError:
            # Если файл не находится внутри base_dir, используем альтернативный подход
            return self._generate_import_path_alternative(py_file)

    def _generate_import_path_alternative(self, py_file: Path) -> str:
        """
        Альтернативный метод генерации import path
        """
        # Ищем папку custom_nodes в пути
        path_parts = list(py_file.parts)
        
        try:
            custom_nodes_index = path_parts.index('custom_nodes')
            # Берем все части после custom_nodes
            relevant_parts = path_parts[custom_nodes_index + 1:]
            
            # Убираем расширение .py
            if relevant_parts[-1].endswith('.py'):
                relevant_parts[-1] = relevant_parts[-1][:-3]
            
            # Убираем __init__ если есть
            if relevant_parts and relevant_parts[-1] == '__init__':
                relevant_parts = relevant_parts[:-1]
            
            return 'custom_nodes.' + '.'.join(relevant_parts)
            
        except ValueError:
            # Если не нашли custom_nodes, используем имя файла
            return py_file.stem

    def _find_node_class_mappings_in_ast(self, tree: ast.AST) -> Dict[str, str]:
        """
        Ищет NODE_CLASS_MAPPINGS в AST дереве
        """
        mappings = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'NODE_CLASS_MAPPINGS':
                        if isinstance(node.value, ast.Dict):
                            # Извлекаем ключи и значения из словаря
                            keys = []
                            values = []
                            
                            for key in node.value.keys:
                                if isinstance(key, ast.Str):
                                    keys.append(key.s)
                                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                                    keys.append(key.value)
                            
                            for value in node.value.values:
                                if isinstance(value, ast.Name):
                                    values.append(value.id)
                                elif isinstance(value, ast.Attribute):
                                    # Обрабатываем атрибуты like module.Class
                                    attr_name = []
                                    current = value
                                    while isinstance(current, ast.Attribute):
                                        attr_name.insert(0, current.attr)
                                        current = current.value
                                    if isinstance(current, ast.Name):
                                        attr_name.insert(0, current.id)
                                        values.append('.'.join(attr_name))
                            
                            if len(keys) == len(values):
                                mappings = dict(zip(keys, values))
        
        return mappings

    def _find_imports_with_mappings(self, tree: ast.AST) -> List[Dict]:
        """
        Ищет импорты, которые могут содержать NODE_CLASS_MAPPINGS
        """
        imports = []
        
        for node in ast.walk(tree):
            # Ищем импорты вида: from .module import NODE_CLASS_MAPPINGS
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == 'NODE_CLASS_MAPPINGS':
                        imports.append({
                            'module': node.module,
                            'level': node.level,
                            'name': alias.name
                        })
            
            # Ищем импорты вида: import module (и потом проверяем присваивания)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'level': 0,
                        'name': None
                    })
            
            # Ищем присваивания вида: NODE_CLASS_MAPPINGS = module.NODE_CLASS_MAPPINGS
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'NODE_CLASS_MAPPINGS':
                        if isinstance(node.value, ast.Attribute):
                            # Пример: NODE_CLASS_MAPPINGS = module.mapping
                            attr = node.value
                            if isinstance(attr.value, ast.Name):
                                imports.append({
                                    'module': attr.value.id,
                                    'level': 0,
                                    'name': attr.attr
                                })
        
        return imports

    def _find_imported_file(self, import_info: Dict, base_dir: Path, current_file: Path) -> Optional[Path]:
        """
        Находит файл на основе информации об импорте
        """
        try:
            module_name = import_info['module']
            level = import_info.get('level', 0)
            
            if level > 0:  # Относительный импорт (from .module import ...)
                if level == 1:
                    target_dir = current_file.parent
                else:
                    target_dir = current_file.parent
                    for _ in range(level - 1):
                        target_dir = target_dir.parent
                
                # Пробуем разные варианты имени файла
                possible_files = [
                    target_dir / f"{module_name}.py",
                    target_dir / module_name / "__init__.py",
                    target_dir / module_name / f"{module_name}.py"
                ]
                
                for file_path in possible_files:
                    if file_path.exists():
                        return file_path
            
            else:  # Абсолютный импорт
                # Пробуем найти в текущей директории
                possible_files = [
                    base_dir / f"{module_name}.py",
                    base_dir / module_name / "__init__.py",
                    base_dir / module_name / f"{module_name}.py"
                ]
                
                for file_path in possible_files:
                    if file_path.exists():
                        return file_path
        
        except Exception as e:
            print(f"    Ошибка при поиске импортированного файла: {e}")
        
        return None

    def _find_class_function_values(self, tree: ast.AST) -> Dict[str, str]:
        """
        Ищет значение переменной FUNCTION в классах
        Возвращает словарь: {class_name: function_value}
        """
        class_functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                function_value = None
                
                # Ищем присваивания внутри класса
                for class_item in node.body:
                    if isinstance(class_item, ast.Assign):
                        # Проверяем, присваивается ли переменная FUNCTION
                        for target in class_item.targets:
                            if isinstance(target, ast.Name) and target.id == 'FUNCTION':
                                # Извлекаем значение
                                if isinstance(class_item.value, ast.Str):
                                    function_value = class_item.value.s
                                elif isinstance(class_item.value, ast.Constant) and isinstance(class_item.value.value, str):
                                    function_value = class_item.value.value
                                elif isinstance(class_item.value, ast.Name):
                                    function_value = class_item.value.id
                                break
                
                if function_value is not None:
                    class_functions[node.name] = function_value
        
        return class_functions

    def run_analysis(self) -> List[NodeMappingResult]:
        """
        Запускает полный анализ нодов
        """
        comfyui_path = self.find_comfyui()
        additional_path = self.parse_extra_model_paths(comfyui_path)
        
        all_paths = []
        standard_path = comfyui_path / "custom_nodes"
        all_paths.append(standard_path)
        
        if additional_path:
            for path_value in additional_path:
                path_obj = Path(path_value)
                all_paths.append(path_obj)
        
        all_nodes = self.get_all_subdirectories(all_paths)
        all_nodes.append(comfyui_path)
        extras_path=comfyui_path / "comfy_extras"
        all_nodes.append(extras_path)
        return self.find_node_class_mappings_with_imports(all_nodes)


if __name__ == "__main__":
    analyzer = NodeMappingAnalyzer()
    nodes = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("НАЙДЕННЫЕ НОДЫ:")
    print("="*50)
    
    for node in nodes:
        print(f"\nНода: {node.node_name}")
        print(f"Класс: {node.class_name}")
        print(f"FUNCTION: {node.function_value}")
        print(f"Файл: {node.file_path}")
        print(f"Импорт path: {node.import_path}")
        print(f"Импорт: {node.import_from}")
        print("-" * 30)
    
    print(f"\nВсего найдено нодов: {len(nodes)}")