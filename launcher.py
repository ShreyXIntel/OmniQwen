"""
Enhanced Launcher for the Game Benchmarking System with game-agnostic state management.
Now supports flow-aware navigation that works across different games.
"""
import os
import sys
import logging
import argparse
import time
from typing import Optional

def setup_logging() -> logging.Logger:
    """Set up logging for the enhanced launcher."""
    logging.basicConfig(
        level=logging.DEBUG,  # Always debug mode for enhanced version
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("EnhancedLauncher")

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
            logger.info(f"[OK] {display_name} available")
        except ImportError as e:
            logger.error(f"[MISSING] {display_name} missing: {e}")
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

def verify_enhanced_flow_config(logger: logging.Logger) -> bool:
    """Verify the enhanced YAML flow configuration."""
    flow_file = "game_flows.yaml"
    
    if os.path.exists(flow_file):
        logger.info(f"[OK] Enhanced flow configuration found: {flow_file}")
        
        try:
            import yaml
            with open(flow_file, 'r') as f:
                config = yaml.safe_load(f)
            
            games = config.get('games', {})
            logger.info(f"[OK] YAML is valid with {len(games)} game configurations")
            
            # Verify enhanced features in each game
            enhanced_games = 0
            for game_name, game_config in games.items():
                game_title = game_config.get('name', game_name)
                
                # Check for enhanced features
                has_state_keywords = bool(game_config.get('state_keywords', {}))
                has_flow_states = bool(game_config.get('flow_states', {}))
                has_priority_keywords = bool(game_config.get('priority_keywords', {}))
                
                if has_state_keywords and has_flow_states and has_priority_keywords:
                    enhanced_games += 1
                    logger.info(f"  - {game_name} ({game_title}) [ENHANCED]")
                    
                    # Log enhanced feature details
                    state_count = len(game_config.get('state_keywords', {}))
                    flow_sequence = game_config.get('flow_states', {}).get('sequence', [])
                    priority_count = len(game_config.get('priority_keywords', {}))
                    
                    logger.info(f"    * State Keywords: {state_count} states")
                    logger.info(f"    * Flow Sequence: {len(flow_sequence)} steps ({' -> '.join(flow_sequence[:3])}...)")
                    logger.info(f"    * Priority Keywords: {priority_count} categories")
                else:
                    logger.warning(f"  - {game_name} ({game_title}) [LEGACY - missing enhanced features]")
            
            logger.info(f"Enhanced games: {enhanced_games}/{len(games)}")
            
            if enhanced_games == 0:
                logger.error("No games have enhanced flow configurations!")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced YAML file is invalid: {e}")
            return False
    else:
        logger.error(f"[MISSING] Enhanced flow configuration not found: {flow_file}")
        logger.error("  Please ensure you have the enhanced game_flows.yaml file")
        return False

def verify_enhanced_config(logger: logging.Logger) -> bool:
    """Verify enhanced configuration file."""
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
                logger.info(f"[OK] {config_name} found in config")
            else:
                logger.error(f"[MISSING] {config_name} missing from config")
                return False
        
        # Test enhanced directory creation
        test_dirs = config.create_run_directory("test_enhanced_game")
        logger.info(f"[OK] Enhanced configuration valid, test directories created at {test_dirs['root']}")
        
        # Verify enhanced debug directories
        enhanced_dirs = [
            "omniparser_output",
            "qwen_prompt", 
            "qwen_responses",
            "error_logs"
        ]
        
        for dir_name in enhanced_dirs:
            if dir_name in test_dirs:
                logger.info(f"[OK] Enhanced debug directory: {dir_name}")
            else:
                logger.warning(f"[MISSING] Enhanced debug directory: {dir_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Enhanced configuration error: {e}")
        return False

def test_enhanced_flow_executor(logger: logging.Logger) -> bool:
    """Test the enhanced adaptive flow executor."""
    try:
        from adaptive_flow_executor import AdaptiveFlowExecutor
        
        logger.info("[TESTING] Enhanced Adaptive Flow Executor...")
        
        executor = AdaptiveFlowExecutor("game_flows.yaml")
        
        # Test game loading
        available_games = list(executor.flows.keys())
        if not available_games:
            logger.error("[FAILED] No games found in flow executor")
            return False
        
        logger.info(f"[OK] Found {len(available_games)} games")
        
        # Test enhanced features with first available game
        test_game = available_games[0]
        if executor.set_game(test_game):
            logger.info(f"[OK] Successfully set test game: {test_game}")
            
            # Test enhanced features
            state_keywords = executor.state_keywords
            priority_keywords = executor.priority_keywords
            flow_states = executor.flow_states
            
            logger.info(f"[OK] Loaded {len(state_keywords)} state keyword sets")
            logger.info(f"[OK] Loaded {len(priority_keywords)} priority keyword categories")
            logger.info(f"[OK] Flow states configured: {bool(flow_states)}")
            
            if flow_states:
                sequence = flow_states.get('sequence', [])
                goal_state = flow_states.get('goal_state', 'unknown')
                logger.info(f"[OK] Flow sequence: {len(sequence)} steps, goal: {goal_state}")
            
            return True
        else:
            logger.error(f"[FAILED] Could not set test game: {test_game}")
            return False
            
    except Exception as e:
        logger.error(f"[FAILED] Enhanced flow executor test failed: {e}")
        return False

def run_enhanced_verification_tests(logger: logging.Logger) -> bool:
    """Run all enhanced verification tests."""
    logger.info("Running Enhanced System Verification Tests...")
    
    tests = [
        ("Dependencies", verify_dependencies),
        ("Enhanced Flow Configuration", verify_enhanced_flow_config),
        ("Enhanced Configuration", verify_enhanced_config),
        ("Enhanced Flow Executor", test_enhanced_flow_executor)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            results[test_name] = test_func(logger)
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Enhanced summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED VERIFICATION SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name:30} : {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("[SUCCESS] All enhanced verification tests passed!")
        logger.info("Features verified:")
        logger.info("  - Game-agnostic state detection from YAML")
        logger.info("  - Flow progression awareness")
        logger.info("  - Enhanced debug output saving")
        logger.info("  - Adaptive state machine with context")
        return True
    else:
        logger.error("[FAILED] Some enhanced verification tests failed")
        logger.error("Please resolve the issues before running the enhanced benchmarker")
        return False

def show_enhanced_game_info(logger: logging.Logger, game_name: str):
    """Show detailed information about a game's enhanced configuration."""
    try:
        from flow_based_benchmarker import FlowBasedGameBenchmarker
        benchmarker = FlowBasedGameBenchmarker()
        
        game_info = benchmarker.get_game_flow_info(game_name)
        if not game_info:
            logger.error(f"Game '{game_name}' not found")
            return
        
        logger.info(f"\n=== ENHANCED GAME INFO: {game_name.upper()} ===")
        logger.info(f"Name: {game_info.get('name', game_name)}")
        logger.info(f"Benchmark Duration: {game_info.get('benchmark_duration', 'default')}s")
        logger.info(f"Sleep Time: {game_info.get('benchmark_sleep_time', 'default')}s")
        
        # Enhanced info
        enhanced_info = game_info.get('enhanced_info', {})
        logger.info(f"\n=== ENHANCED FEATURES ===")
        logger.info(f"Flow Steps: {enhanced_info.get('total_flow_steps', 0)}")
        logger.info(f"State Keywords: {enhanced_info.get('state_keywords_count', 0)} states")
        logger.info(f"Priority Keywords: {enhanced_info.get('priority_keywords_count', 0)} categories")
        logger.info(f"Goal State: {enhanced_info.get('goal_state', 'unknown')}")
        
        # Flow sequence
        flow_states = game_info.get('flow_states', {})
        if flow_states:
            sequence = flow_states.get('sequence', [])
            logger.info(f"\n=== FLOW SEQUENCE ===")
            logger.info(f"Expected flow: {' -> '.join(sequence)}")
            
            transitions = flow_states.get('transitions', {})
            if transitions:
                logger.info(f"\n=== STATE TRANSITIONS ===")
                for state, next_states in list(transitions.items())[:5]:  # Show first 5
                    logger.info(f"  {state} -> {next_states}")
        
        # Priority keywords
        priority_keywords = game_info.get('priority_keywords', {})
        if priority_keywords:
            logger.info(f"\n=== PRIORITY KEYWORDS ===")
            for category, keywords in priority_keywords.items():
                logger.info(f"  {category}: {keywords}")
        
        # Legacy flow (if exists)
        legacy_flow = game_info.get('flow', [])
        if legacy_flow:
            logger.info(f"\n=== LEGACY FLOW STEPS ===")
            for step in legacy_flow[:3]:  # Show first 3 steps
                logger.info(f"  Step {step.get('step', '?')}: {step.get('description', 'No description')}")
        
    except Exception as e:
        logger.error(f"Error showing game info: {e}")

def main():
    """Main entry point for the enhanced launcher."""
    parser = argparse.ArgumentParser(
        description='Enhanced Game Benchmarker with Game-Agnostic State Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED FEATURES:
  - Game-agnostic state detection from YAML configuration
  - Flow progression awareness for intelligent navigation
  - Comprehensive debug logging with bulletproof output saving
  - Adaptive state machine that works across different games
  - Qwen 2.5 with flow-aware context understanding

NO LAUNCH VERSION - Always assumes game is running!

Enhanced capabilities:
  - Works with any game that has YAML configuration
  - Understands flow progression and expected next states
  - Saves comprehensive debug outputs for troubleshooting
  - Adapts to different game UI patterns automatically

Examples:
  python launcher.py --verify                    # Run enhanced verification tests
  python launcher.py --game far_cry_6            # Run Far Cry 6 with enhanced features
  python launcher.py --game cyberpunk_2077       # Run Cyberpunk 2077 with flow awareness
  python launcher.py --game counter_strike_2     # Run CS2 with game-specific detection
  python launcher.py --list-games                # Show all games with enhanced info
  python launcher.py --game-info far_cry_6       # Show detailed game configuration

IMPORTANT: Make sure your game is running and at the main menu before starting!
        """
    )
    
    parser.add_argument('--verify', action='store_true', 
                       help='Run enhanced verification tests and exit')
    parser.add_argument('--game', type=str, 
                       help='Game name from enhanced config (e.g., far_cry_6, cyberpunk_2077)')
    parser.add_argument('--list-games', action='store_true',
                       help='List available games with enhanced info and exit')
    parser.add_argument('--game-info', type=str,
                       help='Show detailed information about a specific game')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Benchmark timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Setup logging (always debug mode for enhanced version)
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("  ENHANCED AUTOMATED GAME BENCHMARKER")
    logger.info("  Game-Agnostic State Management + Flow Awareness")
    logger.info("  OmniParser V2 + Qwen 2.5 7B Instruct")
    logger.info("="*70)
    logger.info("Enhanced Features:")
    logger.info("  - Game-agnostic state detection from YAML")
    logger.info("  - Flow progression awareness")
    logger.info("  - Comprehensive debug logging")
    logger.info("  - Adaptive state machine")
    logger.info("  - Qwen with flow context")
    logger.info("  - Works across different game UIs")
    logger.info("="*70)
    
    # Run enhanced verification tests
    if args.verify:
        if not run_enhanced_verification_tests(logger):
            sys.exit(1)
        logger.info("Enhanced verification complete. Exiting.")
        sys.exit(0)
    
    # Show detailed game info
    if args.game_info:
        show_enhanced_game_info(logger, args.game_info)
        sys.exit(0)
    
    # List available games with enhanced info
    if args.list_games:
        try:
            from flow_based_benchmarker import FlowBasedGameBenchmarker
            benchmarker = FlowBasedGameBenchmarker()
            available_games = benchmarker.get_available_games()
            
            logger.info("AVAILABLE GAMES WITH ENHANCED FEATURES:")
            logger.info("="*50)
            
            enhanced_count = 0
            legacy_count = 0
            
            for game_name in available_games:
                game_info = benchmarker.get_game_flow_info(game_name)
                if game_info:
                    game_title = game_info.get('name', game_name)
                    enhanced_info = game_info.get('enhanced_info', {})
                    
                    has_enhanced = (
                        enhanced_info.get('has_flow_states', False) and
                        enhanced_info.get('state_keywords_count', 0) > 0 and
                        enhanced_info.get('priority_keywords_count', 0) > 0
                    )
                    
                    if has_enhanced:
                        enhanced_count += 1
                        status = "[ENHANCED]"
                        flow_steps = enhanced_info.get('total_flow_steps', 0)
                        state_keywords = enhanced_info.get('state_keywords_count', 0)
                        logger.info(f"  - {game_name} ({game_title}) {status}")
                        logger.info(f"    * Flow: {flow_steps} steps, States: {state_keywords} keywords")
                    else:
                        legacy_count += 1
                        status = "[LEGACY]"
                        legacy_steps = len(game_info.get('flow', []))
                        logger.info(f"  - {game_name} ({game_title}) {status}")
                        logger.info(f"    * Legacy flow: {legacy_steps} steps")
                else:
                    logger.info(f"  - {game_name} [ERROR - no config]")
            
            logger.info("="*50)
            logger.info(f"Total: {len(available_games)} games")
            logger.info(f"Enhanced: {enhanced_count} games")
            logger.info(f"Legacy: {legacy_count} games")
            logger.info("\nUse --game-info <game_name> for detailed information")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error listing games: {e}")
            sys.exit(1)
    
    # Quick dependency check
    logger.info("Performing quick dependency check...")
    if not verify_dependencies(logger):
        logger.error("Dependency check failed. Run with --verify for detailed information.")
        sys.exit(1)
    
    # Validate game parameter
    if not args.game:
        logger.error("No game specified! Use --game <name> to specify which game to benchmark.")
        logger.info("Use --list-games to see available options.")
        logger.info("Example: python launcher.py --game far_cry_6")
        sys.exit(1)
    
    # Initialize and run enhanced benchmarker
    try:
        from flow_based_benchmarker import FlowBasedGameBenchmarker
        
        # Apply configuration overrides
        logger.info(f"Setting benchmark timeout to {args.timeout} seconds")
        import config
        config.BENCHMARK_CONFIG["benchmark_timeout"] = args.timeout
        config.BENCHMARK_CONFIG["benchmark_duration"] = min(args.timeout - 20, args.timeout)
        
        # Force enable all debug options for enhanced version
        config.DEBUG_CONFIG["enabled"] = True
        config.DEBUG_CONFIG["verbose_logging"] = True
        config.DEBUG_CONFIG["save_screenshots"] = True
        config.DEBUG_CONFIG["save_omniparser_outputs"] = True
        config.DEBUG_CONFIG["save_qwen_responses"] = True
        
        # Initialize enhanced flow-based benchmarker
        logger.info("Initializing Enhanced Flow-Based Game Benchmarker...")
        benchmarker = FlowBasedGameBenchmarker()
        
        # Show available games
        available_games = benchmarker.get_available_games()
        logger.info(f"Available games: {', '.join(available_games)}")
        
        # Validate game
        game_name = args.game
        if game_name not in available_games:
            logger.error(f"Game '{game_name}' not found in enhanced flow configuration")
            logger.info(f"Available games: {', '.join(available_games)}")
            sys.exit(1)
        
        # Enhanced validation
        if not benchmarker.validate_game_flow(game_name):
            logger.error(f"Invalid or incomplete enhanced flow configuration for game: {game_name}")
            logger.info("Use --game-info <game_name> to see what's missing")
            sys.exit(1)
        
        # Show enhanced game flow info
        logger.info("="*50)
        show_enhanced_game_info(logger, game_name)
        logger.info("="*50)
        
        # IMPORTANT WARNING
        logger.warning("=" * 70)
        logger.warning("  IMPORTANT: MAKE SURE YOUR GAME IS RUNNING!")
        logger.warning("  This enhanced version assumes the game is already at the main menu.")
        logger.warning("  The system will intelligently detect your current state and navigate.")
        logger.warning("  Press Ctrl+C to cancel if your game is not running.")
        logger.warning("=" * 70)
        
        # Give user a chance to cancel
        for i in range(5, 0, -1):
            logger.info(f"Starting enhanced benchmark in {i} seconds... (Ctrl+C to cancel)")
            time.sleep(1)
        
        # Run the enhanced benchmark
        logger.info(f"Starting enhanced benchmark for {game_name}...")
        logger.info("Enhanced features active:")
        logger.info("  - Game-agnostic state detection")
        logger.info("  - Flow progression awareness")
        logger.info("  - Comprehensive debug logging")
        logger.info("  - Adaptive navigation")
        
        success = benchmarker.run_benchmark_for_game(
            game_name=game_name,
            launch_game=False,  # NEVER launch
            game_path=None      # Not needed
        )
        
        if success:
            logger.info("[SUCCESS] Enhanced benchmark completed successfully!")
            logger.info("[INFO] Check the benchmark_runs/ directory for comprehensive results")
            logger.info("[INFO] All debug outputs saved with enhanced features")
            logger.info("[INFO] Flow progression and state detection logs available")
        else:
            logger.error("[FAILED] Enhanced benchmark failed to complete successfully")
            logger.info("[INFO] Check enhanced debug outputs in benchmark_runs/ for troubleshooting")
            logger.info("[INFO] Flow execution logs will show where the issue occurred")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPTED] Enhanced benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Error during enhanced benchmark execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()