#!/usr/bin/env python3
"""
OpenRouter Models Utility Script

Fetches model information from OpenRouter API and generates YAML configuration files
for use with Continue 1.0 blocks.

This version adds:
1. Version tracking for each model (incremented when model configuration changes)
2. Change detection to update YAML files only when necessary
3. Summary reporting of added/modified/unchanged models

Usage:
    python openrouter_models.py [options]

Options:
    --api-key KEY     OpenRouter API key (can also use OPENROUTER_API_KEY env var)
    --input-file FILE Input JSON file with OpenRouter models data (optional)
    --output-dir DIR  Output directory for YAML files (default: ./blocks/public)
    --skip-free       Skip free models (models with zero pricing)
    --summary         Print summary statistics
    --help            Show this help message and exit
"""

import argparse
import json
import os
import re
import sys
import yaml
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
import requests
import logging
from pathlib import Path
import semver


# Set up logging to stdout only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_OUTPUT_DIR = "./blocks/public"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
VERSION_CACHE_FILE = ".version_cache.json"

# Curated model map - only these models will be included
# Format: "openrouter_model_id": "filename-slug"
# Display names are auto-extracted from the API (text after first colon, if present)
# This allows you to:
# 1. Curate which models to include
# 2. Control the generated YAML filenames (slugs)
CURATED_MODELS = {
    # OpenAI Models
    # "openai/gpt-4o": "gpt-4o",
    # "openai/gpt-4o-mini": "gpt-4o-mini", 
    # "openai/gpt-4-turbo": "gpt-4-turbo",
    # "openai/gpt-4": "gpt-4",
    # "openai/gpt-3.5-turbo": "gpt-3.5-turbo",
    # "openai/o1-mini": "o1-mini",
    
    # Anthropic Models
    
    # Meta Llama Models
    
    # Google Models
    
    # Mistral Models
    # "mistralai/mistral-large": "mistral-large",
    # "mistralai/mistral-nemo": "mistral-nemo",
    # "mistralai/codestral": "codestral",
    # "mistralai/mistral-7b-instruct": "mistral-7b",
    
    # DeepSeek Models
    # "deepseek/deepseek-chat": "deepseek-chat",
    # "deepseek/deepseek-coder": "deepseek-coder",
    # "deepseek/deepseek-r1": "deepseek-r1",
    
    # Qwen Models
    "qwen/qwen3-coder": "qwen3-coder",
    
    # Moonshot Models
    "moonshotai/kimi-k2": "kimi-k2",
    
    # Add more models as needed...
}

# Models that should be assigned the autocomplete role
# These will be matched against the auto-extracted display names from the API
AUTOCOMPLETE_MODELS = [
    "Llama 3.1 8B Instruct",
    "Llama 3.2 3B Instruct", 
    "Llama 3.2 1B Instruct",
    "Gemma 2 9B IT",
    "Mistral 7B Instruct",
    "Codestral",
    "Qwen 2.5 7B Instruct",
]

# Models that should have the tool_use capability
# These models can handle function calling and tool use protocols
TOOL_USE_MODELS = [
    # Any model containing these substrings will receive the tool_use capability
    "llama",
    "gemma",
    "gemini",
    "deepseek",
    "mistral",
    "qwen",
    "claude",
    "gpt",
    "o1",
    "kimi"
]

def sanitize_filename(name):
    """Convert the model name to a safe filename."""
    # Replace spaces with hyphens
    result = name.lower().replace(' ', '-')

    # Remove brackets, braces, and parentheses without replacement
    result = re.sub(r'[\[\]{}()]', '', result)

    # Replace any remaining unsafe characters with underscores
    # Keep hyphens (-) and underscores (_) intact
    result = re.sub(r'[^\w\-\.]', '_', result)

    # Reduce repeated underscores or hyphens to single instances
    result = re.sub(r'_+', '_', result)
    result = re.sub(r'-+', '-', result)

    # Strip trailing hyphens or underscores
    result = result.rstrip('-_')

    return result


def has_tool_use_capability(model_id, model_name):
    """Check if a model should have the tool_use capability."""
    model_id_lower = model_id.lower()
    model_name_lower = model_name.lower()
    
    for model_pattern in TOOL_USE_MODELS:
        if (model_pattern.lower() in model_id_lower or 
            model_pattern.lower() in model_name_lower):
            return True
    
    return False

def determine_roles_and_capabilities(model_data, curated_name):
    """Determine appropriate roles and capabilities based on model type."""
    roles = []
    capabilities = []
    
    # Get model info
    model_id = model_data.get('id', '')
    context_length = model_data.get('context_length', 0)
    architecture = model_data.get('architecture', {})
    modality = architecture.get('modality', '')
    
    # Determine roles based on modality and context length
    if 'text->text' in modality:
        roles = ['chat']
        
        # Models with larger context are better for complex tasks like 'apply' and 'edit'
        if context_length >= 8192:
            roles.extend(['apply', 'edit'])
    
    # Add autocomplete role based on the predefined list
    # Use the curated name for matching
    if curated_name in AUTOCOMPLETE_MODELS:
        if 'autocomplete' not in roles:
            roles.append('autocomplete')
    
    # Check if this model should have the tool_use capability
    if has_tool_use_capability(model_id, curated_name):
        capabilities.append('tool_use')
    
    # Remove duplicates and return sorted lists
    return sorted(list(set(roles))), sorted(list(set(capabilities)))

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def load_version_cache():
    """Load version cache from file if it exists, otherwise return empty dict."""
    if os.path.exists(VERSION_CACHE_FILE):
        try:
            with open(VERSION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing version cache file: {e}. Creating new cache.")
            return {}
    return {}

def save_version_cache(cache):
    """Save version cache to file."""
    with open(VERSION_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def validate_yaml_content(yaml_content):
    """Validate that the YAML content meets Continue requirements."""
    errors = []
    
    # Required top-level fields
    required_fields = ['name', 'version', 'schema', 'models']
    for field in required_fields:
        if field not in yaml_content:
            errors.append(f"Missing required top-level field: {field}")
    
    # Check schema version
    if yaml_content.get('schema') != 'v1':
        errors.append(f"Invalid schema version: {yaml_content.get('schema')} (expected 'v1')")
    
    # Check models array
    models = yaml_content.get('models', [])
    if not models or not isinstance(models, list):
        errors.append("'models' must be a non-empty array")
    else:
        for i, model in enumerate(models):
            # Required model fields
            model_required_fields = ['name', 'provider', 'model', 'apiKey']
            for field in model_required_fields:
                if field not in model:
                    errors.append(f"Model #{i+1}: Missing required field: {field}")
            
            # Provider should be 'openrouter'
            if model.get('provider') != 'openrouter':
                errors.append(f"Model #{i+1}: Provider should be 'openrouter', got: {model.get('provider')}")
            
            # Roles should be a non-empty array
            roles = model.get('roles', [])
            if not roles or not isinstance(roles, list):
                errors.append(f"Model #{i+1}: 'roles' must be a non-empty array")
    
    # Return None if no errors, or the list of errors
    return errors if errors else None


def generate_model_hash(model_data):
    """Generate a hash of the model data to detect changes."""
    # Create a simplified model data dictionary with only the fields we care about
    simplified_data = {
        'id': model_data.get('id', ''),
        'name': model_data.get('name', ''),
        'context_length': model_data.get('context_length', 0),
        'pricing': model_data.get('pricing', {}),
        'architecture': model_data.get('architecture', {})
    }
    
    # Add capabilities information to the hash
    model_id = model_data.get('id', '')
    model_name = model_data.get('name', '')
    if has_tool_use_capability(model_id, model_name):
        simplified_data['capabilities'] = ['tool_use']
    
    # Convert to string and hash
    data_str = json.dumps(simplified_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def parse_existing_yaml(filepath):
    """Parse existing YAML file to extract current version."""
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            # Extract YAML content between the --- markers
            match = re.search(r'^---\n(.*?)$', content, re.DOTALL | re.MULTILINE)
            if match:
                yaml_content = match.group(1)
                try:
                    data = yaml.safe_load(yaml_content)
                    return data
                except yaml.YAMLError as e:
                    logger.warning(f"Error parsing YAML in {filepath}: {e}")
                    return None
    except (IOError, FileNotFoundError) as e:
        logger.warning(f"Could not read file {filepath}: {e}")
    return None

def increment_version(current_version):
    """Increment the minor version of the semantic version string."""
    try:
        version_info = semver.VersionInfo.parse(current_version)
        return str(version_info.bump_minor())
    except ValueError:
        # If the version is not a valid semver, default to 1.0.0
        logger.warning(f"Invalid version format: {current_version}. Resetting to 1.0.0")
        return "1.0.0"

def extract_display_name(original_name):
    """Extract display name from original name, taking text after first colon if present."""
    if ':' in original_name:
        return original_name.split(':', 1)[1].strip()
    return original_name

def create_yaml_file(model_data, output_dir=DEFAULT_OUTPUT_DIR, version_cache=None):
    """Create a YAML file for a single model with version tracking."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model ID
    model_id = model_data.get('id', '')
    if not model_id:
        return None  # Skip if no model ID
    
    # Check if this model is in our curated list
    if model_id not in CURATED_MODELS:
        return None  # Skip models not in curated list
    
    # Extract display name from API data (after first colon if present)
    original_name = model_data.get('name', '')
    model_name = extract_display_name(original_name)
    
    # Use the curated slug for filename
    model_slug = CURATED_MODELS[model_id]
    
    # Generate filename using the custom slug
    filename = model_slug + '.yaml'
    filepath = os.path.join(output_dir, filename)
    
    # Determine roles and capabilities 
    roles, capabilities = determine_roles_and_capabilities(model_data, model_name)
    
    # Check if file already exists and determine version
    current_hash = generate_model_hash(model_data)
    version = "1.0.0"  # Default version
    status = "created"
    
    if version_cache is None:
        version_cache = {}
    
    # Check if we have this model in our cache
    if model_id in version_cache:
        prev_hash = version_cache[model_id]['hash']
        prev_version = version_cache[model_id]['version']
        
        # If the hash has changed, increment the version
        if current_hash != prev_hash:
            # Check if this is just adding a tool_use capability
            should_use_patch = False
            
            # Check if model should have tool_use capability and it's not in the previous version
            if has_tool_use_capability(model_id, model_name):
                prev_filepath = None
                prev_filename = version_cache[model_id].get('filename')
                
                if prev_filename:
                    prev_filepath = os.path.join(output_dir, prev_filename)
                    
                    if os.path.exists(prev_filepath):
                        prev_yaml = parse_existing_yaml(prev_filepath)
                        if prev_yaml and 'models' in prev_yaml and len(prev_yaml['models']) > 0:
                            prev_model = prev_yaml['models'][0]
                            if 'capabilities' not in prev_model or 'tool_use' not in prev_model.get('capabilities', []):
                                # Only adding tool_use capability, use patch version
                                should_use_patch = True
            
            # Use patch versioning for capability additions, otherwise minor versioning
            if should_use_patch:
                version_info = semver.VersionInfo.parse(prev_version)
                version = str(version_info.bump_patch())
            else:
                version = increment_version(prev_version)
                
            status = "updated"
        else:
            # No changes, keep the same version
            version = prev_version
            status = "unchanged"
    else:
        # New model, start with version 1.0.0
        status = "created"
    
    # Create YAML content with keys in the desired order
    # Since Python 3.7+, regular dictionaries preserve insertion order
    yaml_content = {
        'name': model_name,
        'version': version,
        'schema': 'v1',
        'models': [
            {
                'name': model_name,
                'provider': 'openrouter',
                'model': model_id,
                'apiKey': '${{ inputs.OPENROUTER_API_KEY }}',
            }
        ]
    }
    
    # Add defaultCompletionOptions with contextLength if available
    # contextLength represents the maximum number of tokens the model can process in a single request
    # This is important for Continue to know the model's capabilities and optimize prompt construction
    context_length = model_data.get('context_length', 0)
    if context_length > 0:
        yaml_content['models'][0]['defaultCompletionOptions'] = {
            'contextLength': context_length
        }
    else:
        logger.warning(f"No context_length found for {model_name} ({model_id}), defaultCompletionOptions will be omitted")
    
    # Check if the model should have tool_use capability
    if has_tool_use_capability(model_id, model_name):
        yaml_content['models'][0]['capabilities'] = ['tool_use']
        
    # Add roles last to maintain desired order
    yaml_content['models'][0]['roles'] = roles
    
    # Only write file if it's new or updated
    # Gather change information for updated models
    change_details = {}
    if status == "updated" and model_id in version_cache:
        # Get previous version info if available
        prev_filename = version_cache[model_id].get('filename')
        prev_filepath = os.path.join(output_dir, prev_filename) if prev_filename else None
        
        # Compare roles if the previous file exists
        if prev_filepath and os.path.exists(prev_filepath):
            prev_yaml = parse_existing_yaml(prev_filepath)
            if prev_yaml and 'models' in prev_yaml and len(prev_yaml['models']) > 0:
                prev_model = prev_yaml['models'][0]
                prev_roles = prev_model.get('roles', [])
                
                # Check for role changes
                added_roles = [r for r in roles if r not in prev_roles]
                removed_roles = [r for r in prev_roles if r not in roles]
                
                if added_roles or removed_roles:
                    change_details['roles'] = {
                        'added': added_roles,
                        'removed': removed_roles
                    }
                
                # Check for context length changes
                prev_context_length = None
                if 'defaultCompletionOptions' in prev_model:
                    prev_context_length = prev_model['defaultCompletionOptions'].get('contextLength')
                
                if prev_context_length != context_length and context_length > 0:
                    change_details['contextLength'] = {
                        'old': prev_context_length,
                        'new': context_length
                    }
                
                # Check for capability changes
                prev_capabilities = prev_model.get('capabilities', [])
                if has_tool_use_capability(model_id, model_name) and ('tool_use' not in prev_capabilities):
                    change_details['capabilities'] = {
                        'added': ['tool_use'],
                        'removed': []
                    }
    
    if status != "unchanged":
        # Validate YAML content
        validation_errors = validate_yaml_content(yaml_content)
        if validation_errors:
            logger.error(f"Validation errors for {model_name}:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            logger.error(f"Skipping generation of {filepath}")
            return None
        
        # Write YAML file with frontmatter
        with open(filepath, 'w') as file:
            file.write('---\n')  # Start frontmatter
            # Configure the YAML dumper to use 2-space indentation
            yaml.dump(yaml_content, file, Dumper=IndentDumper, default_flow_style=False, sort_keys=False, indent=2)
        
        logger.info(f"{status.capitalize()} YAML for {model_name} (version {version})")
    
    # Update cache with new hash and version
    version_cache[model_id] = {
        'hash': current_hash,
        'version': version,
        'filename': filename,
        'name': model_name
    }
    
    return filepath, model_name, roles, model_data.get('architecture', {}).get('modality', 'unknown'), status, version, change_details if status == 'updated' else None

def fetch_models_data(api_key):
    """Fetch models data directly from the OpenRouter API."""
    headers = {
        "accept": "application/json"
    }
    
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    
    try:
        logger.info(f"Fetching models data from {OPENROUTER_API_URL}...")
        response = requests.get(OPENROUTER_API_URL, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for non-200 status codes
        data = response.json()
        if 'data' not in data or not isinstance(data['data'], list):
            logger.error(f"Unexpected API response format: expected 'data' field with list, got {type(data)}")
            return None
        return data['data']
    except requests.RequestException as e:
        logger.error(f"Error fetching data from API: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing API response: {e}")
        return None

def main():
    """Main function to parse arguments and generate YAML files."""
    parser = argparse.ArgumentParser(description='Generate YAML files for OpenRouter models')
    parser.add_argument('-f', '--input-file', help='Input JSON file with OpenRouter models data')
    parser.add_argument('-k', '--api-key', help='OpenRouter API key')
    parser.add_argument('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR, 
                        help='Output directory for YAML files')
    parser.add_argument('--skip-free', action='store_true',
                        help='Skip free models (models with zero pricing)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of all YAML files, ignoring version cache')
    
    args = parser.parse_args()
    
    # Determine API key (priority: command line, environment variable)
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    
    # Get models data from API or file
    models_data = None
    
    if args.input_file:
        # Load from file
        try:
            with open(args.input_file, 'r') as file:
                data = json.load(file)
                # Handle both direct array and {"data": [...]} format
                if isinstance(data, dict) and 'data' in data:
                    models_data = data['data']
                elif isinstance(data, list):
                    models_data = data
                else:
                    logger.error("Input file must contain either an array of models or an object with 'data' field")
                    return 1
            logger.info(f"Loaded {len(models_data)} models from {args.input_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading input file: {e}")
            return 1
    else:
        # Fetch from API (API key not required for public endpoint)
        models_data = fetch_models_data(api_key)
        if not models_data:
            logger.error("Failed to fetch models data from API.")
            return 1
        logger.info(f"Successfully fetched {len(models_data)} models from API")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save API response to file for reference if we fetched data
    if not args.input_file:
        output_file = os.path.join(args.output_dir, "openrouter_api_response.json")
        try:
            with open(output_file, 'w') as file:
                json.dump({"data": models_data}, file, indent=2)
            logger.info(f"Saved API response to {output_file}")
        except Exception as e:
            logger.warning(f"Could not save API response to file: {e}")
    
    # Load version cache (unless we're forcing regeneration)
    version_cache = {} if args.force_regenerate else load_version_cache()
    
    # Process models
    created_files = []
    skipped_models = []
    model_status = {
        "created": [],
        "updated": [],
        "unchanged": []
    }
    role_counter = Counter()
    model_types = Counter()
    model_by_role = defaultdict(list)
    
    total_models = len(models_data)
    logger.info(f"Processing {total_models} models...")
    
    for i, model_data in enumerate(models_data, 1):
        if i % 50 == 0 or i == total_models:
            logger.info(f"Progress: {i}/{total_models} models ({i/total_models:.1%})")
        
        # Get model info for filtering
        model_id = model_data.get('id', '')
        original_name = model_data.get('name', '')
        architecture = model_data.get('architecture', {})
        modality = architecture.get('modality', '')
        
        # Skip non-text models
        if 'text->text' not in modality:
            continue
        
        # Skip models not in our curated list
        if model_id not in CURATED_MODELS:
            continue
        
        # Check if it's a free model and user wants to skip them
        if args.skip_free:
            pricing = model_data.get('pricing', {})
            if pricing and float(pricing.get('prompt', 0)) == 0 and float(pricing.get('completion', 0)) == 0:
                display_name = extract_display_name(original_name)
                skipped_models.append(f"{display_name} (free)")
                continue
        
        # Create YAML file
        result = create_yaml_file(model_data, args.output_dir, version_cache)
        if result:
            filepath, name, roles, model_type, status, version, changes = result
            created_files.append((filepath, name))
            model_status[status].append((name, version))
            
            # Store change details for updated models
            if status == "updated" and changes:
                model_status[status][-1] = (name, version, changes)
            
            # Update statistics
            for role in roles:
                role_counter[role] += 1
                model_by_role[role].append(name)
            
            model_types[model_type] += 1
    
    # Save updated version cache
    save_version_cache(version_cache)
    
    # Print summary
    logger.info(f"\nResults:")
    logger.info(f"  Created: {len(model_status['created'])} models")
    logger.info(f"  Updated: {len(model_status['updated'])} models")
    logger.info(f"  Unchanged: {len(model_status['unchanged'])} models")
    logger.info(f"  Skipped: {len(skipped_models)} models")
    
    # Count models with context length info
    context_length_count = sum(1 for model_data in models_data 
                              if model_data.get('context_length', 0) > 0)
    logger.info(f"  Models with contextLength: {context_length_count}")
    
    if skipped_models:
        logger.info(f"Skipped {len(skipped_models)} models")
    
    if args.summary:
        logger.info("\n=== Summary Statistics ===")
        
        logger.info("\nModel types:")
        for model_type, count in model_types.most_common():
            logger.info(f"  {model_type}: {count} models")
        
        logger.info("\nRoles distribution:")
        for role, count in role_counter.most_common():
            logger.info(f"  {role}: {count} models")
            # Always show all models for autocomplete role
            if role == 'autocomplete' or len(model_by_role[role]) <= 5:
                for model in model_by_role[role]:
                    logger.info(f"    - {model}")
            else:
                for model in model_by_role[role][:3]:  # Show first 3
                    logger.info(f"    - {model}")
                logger.info(f"    - ... and {count-3} more")
        
        # Print curated models configuration
        logger.info(f"\nCurated models configuration:")
        logger.info(f"  Total curated models defined: {len(CURATED_MODELS)}")
        logger.info(f"  Models found in API and processed: {len(created_files)}")
        
        # Print autocomplete eligibility statistics
        logger.info("\nAutocomplete configuration:")
        logger.info(f"  Predefined autocomplete models: {len(AUTOCOMPLETE_MODELS)}")
        logger.info("  Models in predefined list:")
        for model in AUTOCOMPLETE_MODELS:
            logger.info(f"    - {model}")
            
        # Check for models in the list that weren't found in the API
        found_models = set(model_by_role.get('autocomplete', []))
        missing_models = [m for m in AUTOCOMPLETE_MODELS if m not in found_models]
        if missing_models:
            logger.info("\n  Warning: The following models from AUTOCOMPLETE_MODELS were not found in the API data:")
            for model in missing_models:
                logger.info(f"    - {model}")
        
        # Print missing curated models
        processed_model_ids = set()
        for model_data in models_data:
            if model_data.get('id', '') in CURATED_MODELS:
                processed_model_ids.add(model_data.get('id', ''))
        
        missing_curated = [f"{model_id} -> {slug}" for model_id, slug in CURATED_MODELS.items() 
                          if model_id not in processed_model_ids]
        if missing_curated:
            logger.info("\n  Warning: The following curated models were not found in the API data:")
            for model in missing_curated:
                logger.info(f"    - {model}")
        
        # Print added/updated models
        if model_status['created']:
            logger.info("\nNewly added models:")
            for model, version in model_status['created']:
                logger.info(f"  - {model} (v{version})")
        
        if model_status['updated']:
            logger.info("\nUpdated models:")
            for item in model_status['updated']:
                if len(item) == 3:  # We have change details
                    model, version, changes = item
                    logger.info(f"  - {model} (v{version})")
                    
                    # Print role changes
                    if 'roles' in changes:
                        if changes['roles']['added']:
                            logger.info(f"    - Added roles: {', '.join(changes['roles']['added'])}")
                        if changes['roles']['removed']:
                            logger.info(f"    - Removed roles: {', '.join(changes['roles']['removed'])}")
                    
                    # Print context length changes
                    if 'contextLength' in changes:
                        old = changes['contextLength']['old'] or 'none'
                        new = changes['contextLength']['new']
                        logger.info(f"    - Context length: {old} → {new}")
                    
                    # Print capability changes
                    if 'capabilities' in changes:
                        if changes['capabilities']['added']:
                            logger.info(f"    - Added capabilities: {', '.join(changes['capabilities']['added'])}")
                        if changes['capabilities']['removed']:
                            logger.info(f"    - Removed capabilities: {', '.join(changes['capabilities']['removed'])}")
                else:
                    model, version = item
                    logger.info(f"  - {model} (v{version})")

    
    return 0


if __name__ == "__main__":
    sys.exit(main())