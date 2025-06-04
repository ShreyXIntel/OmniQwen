"""
Launcher module for the Game Benchmarking System with OmniParser V2 + Qwen 2.5.
"""
import os
import sys
import logging
import argparse
import time
from typing import Optional

def setup_logging() -> logging.Logger:
    """Set up logging for the launcher."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("Launcher")

def verify_dependencies(logger: logging.Logger) -> bool:
    """Verify that all required dependencies are installed."""
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("PIL", "Pillow"),
        ("win32api", "pywin32"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML")
    ]
    
    missing_modules = []
    
    for module_name, display_name in required_modules:
        try:
            if module_name == "win32api":
                import win32api
                import win32con
                import win32gui
            elif module_name == "PIL":
                from PIL import Image, ImageGrab
            else:
                __import__(module_name)
            logger.info(f"✓ {display_name} available")
        except ImportError as e:
            logger.error(f"✗ {display_name} missing: {e}")
            missing_modules.append(display_name)
    
    if missing_modules:
        logger.error("Missing dependencies. Please install:")
        for module in missing_modules:
            if module == "pywin32":
                logger.error("  pip install pywin32")
            elif module == "Pillow":
                logger.error("  pip install Pillow")
            else:
                logger.error(f"  pip install {module.lower()}")
        return False
    
    return True

def verify_omniparser_dependencies(logger: logging.Logger) -> bool:
    """Verify OmniParser V2 specific dependencies."""
    try:
        # Check for OmniParser utilities
        from util.utils import (
            get_som_labeled_img,
            check_ocr_box,
            get_caption_model_processor,
            get_yolo_model,
        )
        from ultralytics import YOLO
        logger.info("✓ OmniParser V2 utilities available")
        return True
    except ImportError as e:
        logger.error(f"✗ OmniParser V2 dependencies missing: {e}")
        logger.error("Please ensure OmniParser V2 is properly installed and util/ directory is available")
        return False

def verify_gpu(logger: logging.Logger) -> bool:
    """Verify GPU availability for CUDA operations."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            logger.info(f"✓ CUDA available: {device_count} device(s)")
            logger.info(f"  Current device: {current_device} ({device_name})")
            logger.info(f"  Memory: {memory_gb:.1f} GB")
            
            # Test GPU operations
            test_tensor = torch.tensor([1, 2, 3], device="cuda")
            logger.info(f"  GPU test successful: {test_tensor.device}")
            
            return True
        else:
            logger.warning("⚠ CUDA is not available")
            logger.warning("  The system will run on CPU, which may be significantly slower")
            logger.warning("  GPU is highly recommended for OmniParser V2 and Qwen 2.5")
            return False
    except Exception as e:
        logger.error(f"✗ Error testing GPU: {e}")
        return False

def verify_model_files(logger: logging.Logger) -> bool:
    """Verify that required model files exist."""
    model_files_to_check = [
        ("weights/icon_detect/model.pt", "OmniParser icon detection model"),
        ("weights/icon_caption_florence", "OmniParser caption model directory")
    ]
    
    missing_files = []
    
    for file_path, description in model_files_to_check:
        if os.path.exists(file_path):
            logger.info(f"✓ {description} found at {file_path}")
        else:
            logger.error(f"✗ {description} not found at {file_path}")
            missing_files.append((file_path, description))
    
    if missing_files:
        logger.error("Missing model files. Please ensure OmniParser V2 models are downloaded:")
        for file_path, description in missing_files:
            logger.error(f"  {description}: {file_path}")
        return False
    
    return True

def verify_flow_config(logger: logging.Logger) -> bool:
    """Verify that the YAML flow configuration file exists."""
    flow_file = "game_flows.yaml"
    
    if os.path.exists(flow_file):
        logger.info(f"✓ Flow configuration found: {flow_file}")
        
        # Try to load and validate YAML
        try:
            import yaml
            with open(flow_file, 'r') as f:
                config = yaml.safe_load(f)
            
            games = config.get('games', {})
            logger.info(f"✓ YAML is valid with {len(games)} game configurations")
            return True
            
        except Exception as e:
            logger.error(f"✗ YAML file is invalid: {e}")
            return False
    else:
        logger.error(f"✗ Flow configuration not found: {flow_file}")
        logger.error("  Please create the game_flows.yaml file with game configurations")
        return False

def verify_config(logger: logging.Logger) -> bool:
    """Verify configuration file exists and is valid."""
    try:
        import config
        
        # Check required configuration sections
        required_configs = [
            "OMNIPARSER_CONFIG",
            "QWEN_CONFIG", 
            "BENCHMARK_CONFIG",
            "DEBUG_CONFIG"
        ]
        
        for config_name in required_configs:
            if hasattr(config, config_name):
                logger.info(f"✓ {config_name} found in config")
            else:
                logger.error(f"✗ {config_name} missing from config")
                return False
        
        # Test directory creation
        test_dirs = config.create_run_directory()
        logger.info(f"✓ Configuration valid, test directories created at {test_dirs['root']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration error: {e}")
        return False

def verify_permissions(logger: logging.Logger) -> bool:
    """Verify file system permissions."""
    try:
        # Test directory creation
        test_dir = "test_permissions_temp"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test file creation
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Permission test")
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        
        logger.info("✓ File system permissions verified")
        return True
        
    except Exception as e:
        logger.error(f"✗ Permission error: {e}")
        logger.error("  Please run with appropriate permissions or check disk space")
        return False

def run_verification_tests(logger: logging.Logger) -> bool:
    """Run all verification tests."""
    logger.info("Running system verification tests...")
    
    tests = [
        ("Dependencies", verify_dependencies),
        ("OmniParser Dependencies", verify_omniparser_dependencies),
        ("GPU/CUDA", verify_gpu),
        ("Model Files", verify_model_files),
        ("Configuration", verify_config),
        ("Flow Configuration", verify_flow_config),
        ("Permissions", verify_permissions)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            results[test_name] = test_func(logger)
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name:25} : {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*50)
    
    if all_passed:
        logger.info("✅ All verification tests passed!")
        return True
    else:
        logger.error("❌ Some verification tests failed")
        logger.error("Please resolve the issues before running the benchmarker")
        return False

def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description='Game Benchmarker with OmniParser V2 + Qwen 2.5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --verify                    # Run verification tests only
  python launcher.py --game far_cry_6            # Launch Far Cry 6 and run benchmark
  python launcher.py --game-path "C:\\path\\to\\game.exe"  # Use custom game path
  python launcher.py --no-launch                 # Assume game is already running
        """
    )
    
    parser.add_argument('--verify', action='store_true', 
                       help='Run verification tests and exit')
    parser.add_argument('--game', type=str, 
                       help='Game name from config (e.g., far_cry_6, cyberpunk_2077)')
    parser.add_argument('--game-path', type=str, 
                       help='Direct path to game executable')
    parser.add_argument('--no-launch', action='store_true',
                       help='Skip game launch (assume game is already running)')
    parser.add_argument('--timeout', type=int, 
                       help='Override benchmark timeout in seconds')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    logger.info("="*60)
    logger.info("  AUTOMATED GAME BENCHMARKER")
    logger.info("  OmniParser V2 + Qwen 2.5 7B Instruct")
    logger.info("="*60)
    
    # Run verification tests
    if args.verify or not any([args.game, args.game_path, args.no_launch]):
        if not run_verification_tests(logger):
            sys.exit(1)
        
        if args.verify:
            logger.info("Verification complete. Exiting.")
            sys.exit(0)
        
        logger.info("\nVerification passed. You can now run the benchmarker.")
        logger.info("Example: python launcher.py --game far_cry_6")
        sys.exit(0)
    
    # Quick dependency check
    logger.info("Performing quick dependency check...")
    if not verify_dependencies(logger):
        logger.error("Dependency check failed. Run with --verify for detailed information.")
        sys.exit(1)
    
    # Initialize and run benchmarker
    try:
        from flow_based_benchmarker import FlowBasedGameBenchmarker
        
        # Apply configuration overrides
        if args.timeout:
            logger.info(f"Overriding benchmark timeout to {args.timeout} seconds")
            import config
            config.BENCHMARK_CONFIG["benchmark_timeout"] = args.timeout
            config.BENCHMARK_CONFIG["benchmark_duration"] = min(args.timeout - 20, args.timeout)
        
        # Initialize flow-based benchmarker
        logger.info("Initializing Flow-Based Game Benchmarker...")
        benchmarker = FlowBasedGameBenchmarker()
        
        # Show available games
        available_games = benchmarker.get_available_games()
        logger.info(f"Available games: {', '.join(available_games)}")
        
        # Determine game name
        if args.game:
            game_name = args.game
            if game_name not in available_games:
                logger.error(f"Game '{game_name}' not found in flow configuration")
                logger.info(f"Available games: {', '.join(available_games)}")
                sys.exit(1)
        elif args.game_path:
            # Try to guess game name from path or use generic
            game_name = "generic"  # We'll need to add a generic flow
            logger.warning("Using generic flow for custom game path")
        else:
            logger.error("No game specified. Use --game <name> or --game-path <path>")
            logger.info(f"Available games: {', '.join(available_games)}")
            sys.exit(1)
        
        # Validate game flow
        if not benchmarker.validate_game_flow(game_name):
            logger.error(f"Invalid flow configuration for game: {game_name}")
            sys.exit(1)
        
        # Show game flow info
        flow_info = benchmarker.get_game_flow_info(game_name)
        if flow_info:
            steps = len(flow_info.get('flow', []))
            logger.info(f"Game: {flow_info.get('name', game_name)} ({steps} navigation steps)")
        
        # Run the benchmark
        logger.info(f"Starting benchmark for {game_name}...")
        success = benchmarker.run_benchmark_for_game(
            game_name=game_name,
            launch_game=not args.no_launch,
            game_path=args.game_path
        )
        
        if success:
            logger.info("🎉 Benchmark completed successfully!")
            logger.info("📁 Check the benchmark_runs/ directory for results")
        else:
            logger.error("❌ Benchmark failed to complete successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during benchmark execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()