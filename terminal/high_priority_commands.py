"""
High Priority Commands Integration
Command handlers for the 5 new high-priority features
"""

from rich.console import Console

console = Console()


def register_api_commands(aetherai_instance):
    """Register API client commands with the AetherAI instance"""
    api = aetherai_instance.api_client
    
    commands = {
        # HTTP Methods
        'http get': lambda args: api.get(args[0], args[1] if len(args) > 1 else None),
        'http post': lambda args: api.post(args[0], args[1]) if len(args) >= 2 else "Usage: /http post [url] [json_data]",
        'http put': lambda args: api.put(args[0], args[1]) if len(args) >= 2 else "Usage: /http put [url] [json_data]",
        'http delete': lambda args: api.delete(args[0]),
        'http patch': lambda args: api.patch(args[0], args[1]) if len(args) >= 2 else "Usage: /http patch [url] [json_data]",
        
        # Configuration
        'http headers': lambda args: api.set_headers(args[0]) if args else "Usage: /http headers [key:value,...]",
        'http auth': lambda args: api.set_auth(args[0], args[1]) if len(args) >= 2 else "Usage: /http auth [type] [credentials]",
        'http reset': lambda args: api.reset(),
        
        # Collections
        'http save': lambda args: api.save_collection(args[0], aetherai_instance._http_last_method, 
                                                       aetherai_instance._http_last_url,
                                                       aetherai_instance._http_last_data) if args else "Usage: /http save [name]",
        'http collection list': lambda args: api.list_collections(),
        'http collection run': lambda args: api.run_collection(args[0]) if args else "Usage: /http collection run [name]",
        
        # Testing
        'http test': lambda args: api.test_endpoint(args[0], int(args[1]) if len(args) > 1 else 200),
        'http benchmark': lambda args: api.benchmark(args[0], int(args[1]) if len(args) > 1 else 10),
        
        # History
        'http history': lambda args: api.show_history(),
        'http clear': lambda args: api.clear_history(),
    }
    
    return commands


def register_db_commands(aetherai_instance):
    """Register database commands with the AetherAI instance"""
    db = aetherai_instance.database_manager
    
    commands = {
        # Connection Management
        'db connect': lambda args: db.connect(args[0], args[1], args[2] if len(args) > 2 else "default") if len(args) >= 2 else 
                     "Usage: /db connect [type] [connection_string] [optional:name]\n   Types: sqlite, postgresql, mysql, mongodb",
        'db list': lambda args: db.list_connections(),
        'db close': lambda args: db.close(args[0] if args else None),
        
        # Query Operations
        'db query': lambda args: db.query(' '.join(args)) if args else "Usage: /db query [SQL]",
        'db tables': lambda args: db.show_tables(),
        'db schema': lambda args: db.describe_table(args[0]) if args else "Usage: /db schema [table_name]",
        
        # Backup & Restore
        'db backup': lambda args: db.backup(args[0]) if args else "Usage: /db backup [output_path]",
        'db stats': lambda args: db.stats(),
        'db script': lambda args: db.execute_script(args[0]) if args else "Usage: /db script [file_path]",
    }
    
    return commands


def register_pkg_commands(aetherai_instance):
    """Register package manager commands with the AetherAI instance"""
    pkg = aetherai_instance.package_manager
    
    commands = {
        # Package Operations
        'pkg search': lambda args: pkg.search(args[0], args[1] if len(args) > 1 else None) if args else "Usage: /pkg search [package]",
        'pkg install': lambda args: pkg.install(args[0], args[1] if len(args) > 1 else None) if args else "Usage: /pkg install [package] [optional:pm]",
        'pkg uninstall': lambda args: pkg.uninstall(args[0], args[1] if len(args) > 1 else None) if args else "Usage: /pkg uninstall [package]",
        'pkg update': lambda args: pkg.update(args[0] if args else None),
        
        # Information
        'pkg outdated': lambda args: pkg.list_outdated(),
        'pkg audit': lambda args: pkg.audit(),
        'pkg info': lambda args: pkg.info(args[0]) if args else "Usage: /pkg info [package]",
        'pkg scripts': lambda args: pkg.list_scripts(),
        'pkg clean': lambda args: pkg.clean_cache(),
        'pkg current': lambda args: pkg.get_current_pm(),
    }
    
    return commands


def register_test_commands(aetherai_instance):
    """Register testing commands with the AetherAI instance"""
    test = aetherai_instance.test_runner
    
    commands = {
        # Test Execution
        'test run': lambda args: test.run_tests(args[0] if args else None, args[1] if len(args) > 1 else None, verbose=True),
        'test coverage': lambda args: test.run_coverage(args[0] if args else None),
        'test watch': lambda args: test.watch_mode(args[0] if args else None),
        'test parallel': lambda args: test.run_parallel(int(args[0]) if args else 4),
        
        # Test Information
        'test list': lambda args: test.list_tests(),
        'test report': lambda args: test.generate_report(args[0] if args else 'html'),
        'test results': lambda args: test.get_last_results(),
        'test framework': lambda args: test.get_current_framework(),
        
        # Specific Tests
        'test file': lambda args: test.run_specific(args[0]) if args else "Usage: /test file [path]",
    }
    
    return commands


def register_watch_commands(aetherai_instance):
    """Register file watcher commands with the AetherAI instance"""
    watch = aetherai_instance.file_watcher
    
    commands = {
        # Watch Management
        'watch start': lambda args: watch.start(args[0], ' '.join(args[1:])) if len(args) >= 2 else 
                      "Usage: /watch start [path] [action]",
        'watch stop': lambda args: watch.stop(args[0]) if args else "Usage: /watch stop [id]",
        'watch list': lambda args: watch.list_watchers(),
        'watch logs': lambda args: watch.show_logs(int(args[0]) if args else 20),
        'watch stop-all': lambda args: watch.stop_all(),
        'watch status': lambda args: watch.get_status(),
        
        # Convenience Methods
        'watch compile': lambda args: watch.watch_compile(args[0], args[1] if len(args) > 1 else "auto") if args else 
                        "Usage: /watch compile [path] [optional:compiler]",
        'watch lint': lambda args: watch.watch_lint(args[0], args[1] if len(args) > 1 else "auto") if args else 
                     "Usage: /watch lint [path] [optional:linter]",
        'watch test': lambda args: watch.watch_test(args[0], args[1] if len(args) > 1 else "auto") if args else 
                     "Usage: /watch test [path] [optional:test_cmd]",
        'watch format': lambda args: watch.watch_format(args[0], args[1] if len(args) > 1 else "auto") if args else 
                       "Usage: /watch format [path] [optional:formatter]",
        'watch reload': lambda args: watch.watch_reload(args[0], int(args[1]) if len(args) > 1 else 3000) if args else 
                       "Usage: /watch reload [path] [optional:port]",
    }
    
    return commands


def get_all_high_priority_commands(aetherai_instance):
    """Get all high-priority command handlers"""
    all_commands = {}
    
    # Only register if modules are available
    if aetherai_instance.api_client:
        all_commands.update(register_api_commands(aetherai_instance))
    
    if aetherai_instance.database_manager:
        all_commands.update(register_db_commands(aetherai_instance))
    
    if aetherai_instance.package_manager:
        all_commands.update(register_pkg_commands(aetherai_instance))
    
    if aetherai_instance.test_runner:
        all_commands.update(register_test_commands(aetherai_instance))
    
    if aetherai_instance.file_watcher:
        all_commands.update(register_watch_commands(aetherai_instance))
    
    return all_commands


def get_high_priority_help_text() -> str:
    """Get help text for high-priority features"""
    from rich.text import Text
    
    help_text = Text()
    
    help_text.append("\n🚀 HIGH-PRIORITY FEATURES:\n\n", style="bold magenta")
    
    # API Testing
    help_text.append("🌐 API TESTING & REQUESTS:\n", style="bold yellow")
    help_text.append("/http get [url]                       - Make GET request\n", style="white")
    help_text.append("/http post [url] [json]               - Make POST request\n", style="white")
    help_text.append("/http put [url] [json]                - Make PUT request\n", style="white")
    help_text.append("/http delete [url]                    - Make DELETE request\n", style="white")
    help_text.append("/http headers [key:value,...]         - Set custom headers\n", style="white")
    help_text.append("/http auth [type] [credentials]       - Set authentication\n", style="white")
    help_text.append("/http save [name]                     - Save request to collection\n", style="white")
    help_text.append("/http collection list                 - List saved collections\n", style="white")
    help_text.append("/http collection run [name]           - Run saved request\n", style="white")
    help_text.append("/http test [url] [expected_status]    - Test API endpoint\n", style="white")
    help_text.append("/http benchmark [url] [requests]      - Benchmark API performance\n", style="white")
    help_text.append("/http history                         - Show request history\n\n", style="white")
    
    # Database Management
    help_text.append("🗄️ DATABASE MANAGEMENT:\n", style="bold yellow")
    help_text.append("/db connect [type] [conn_str] [name]  - Connect to database\n", style="white")
    help_text.append("/db list                              - List connections\n", style="white")
    help_text.append("/db query [SQL]                       - Execute SQL query\n", style="white")
    help_text.append("/db tables                            - List all tables\n", style="white")
    help_text.append("/db schema [table]                    - Show table schema\n", style="white")
    help_text.append("/db backup [path]                     - Backup database\n", style="white")
    help_text.append("/db stats                             - Database statistics\n", style="white")
    help_text.append("/db script [file]                     - Execute SQL script\n", style="white")
    help_text.append("/db close [name]                      - Close connection\n\n", style="white")
    
    # Package Manager
    help_text.append("📦 PACKAGE MANAGER:\n", style="bold yellow")
    help_text.append("/pkg search [package]                 - Search for packages\n", style="white")
    help_text.append("/pkg install [package]                - Install package\n", style="white")
    help_text.append("/pkg uninstall [package]              - Uninstall package\n", style="white")
    help_text.append("/pkg update [package]                 - Update package(s)\n", style="white")
    help_text.append("/pkg outdated                         - List outdated packages\n", style="white")
    help_text.append("/pkg audit                            - Security audit\n", style="white")
    help_text.append("/pkg info [package]                   - Package information\n", style="white")
    help_text.append("/pkg scripts                          - List available scripts\n", style="white")
    help_text.append("/pkg clean                            - Clean package cache\n\n", style="white")
    
    # Test Runner
    help_text.append("🧪 TEST AUTOMATION:\n", style="bold yellow")
    help_text.append("/test run [pattern]                   - Run tests\n", style="white")
    help_text.append("/test coverage                        - Run with coverage\n", style="white")
    help_text.append("/test watch                           - Watch mode testing\n", style="white")
    help_text.append("/test parallel [workers]              - Run tests in parallel\n", style="white")
    help_text.append("/test list                            - List all tests\n", style="white")
    help_text.append("/test report [format]                 - Generate test report\n", style="white")
    help_text.append("/test file [path]                     - Run specific test\n\n", style="white")
    
    # File Watcher
    help_text.append("👁️ FILE WATCHER:\n", style="bold yellow")
    help_text.append("/watch start [path] [action]          - Start watching path\n", style="white")
    help_text.append("/watch stop [id]                      - Stop watcher\n", style="white")
    help_text.append("/watch list                           - List active watchers\n", style="white")
    help_text.append("/watch compile [path]                 - Auto-compile on change\n", style="white")
    help_text.append("/watch lint [path]                    - Auto-lint on change\n", style="white")
    help_text.append("/watch test [path]                    - Auto-test on change\n", style="white")
    help_text.append("/watch format [path]                  - Auto-format on change\n", style="white")
    help_text.append("/watch reload [path]                  - Auto-reload browser\n", style="white")
    help_text.append("/watch logs                           - Show watch logs\n\n", style="white")
    
    return help_text
