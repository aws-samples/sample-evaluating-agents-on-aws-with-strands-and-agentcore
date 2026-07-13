# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data ingestion AWS Lambda handler with mocked BigQuery.

Instead of calling BigQuery API, this reads sample vehicle data from Amazon S3,
processes it through the same pipeline (contextualization + embeddings),
and writes LanceDB-formatted data back to Amazon S3.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client("s3")
bedrock_runtime = boto3.client("bedrock-runtime")

# Environment variables
DATA_BUCKET = os.environ["DATA_BUCKET"]
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
MOCK_BIGQUERY = os.environ.get("MOCK_BIGQUERY", "true") == "true"
SAMPLE_DATA_KEY = os.environ.get("SAMPLE_DATA_KEY", "raw/sample_vehicles.json")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda handler for data ingestion.

    Args:
        event: Lambda event (from EventBridge or manual invoke)
        context: Lambda context

    Returns:
        Response with status and metadata
    """
    logger.info("Starting data ingestion for environment: %s", ENVIRONMENT)
    logger.info("Mock BigQuery mode: %s", MOCK_BIGQUERY)

    try:
        # Step 1: Fetch vehicle data (mocked BigQuery)
        if MOCK_BIGQUERY:
            logger.info("Fetching sample vehicle data from S3 (mocked BigQuery)")
            vehicles = fetch_sample_data_from_s3()
        else:
            # Real BigQuery integration would go here
            raise NotImplementedError("Real BigQuery integration not implemented")

        logger.info("Fetched %s vehicles", len(vehicles))

        # Step 2: Contextualize vehicle descriptions with Amazon Nova Lite v1
        logger.info("Contextualizing vehicle descriptions with Amazon Nova Lite v1")
        vehicles_with_context = contextualize_vehicles(vehicles)

        # Step 3: Generate embeddings with Amazon Titan Text Embeddings V2
        logger.info("Generating embeddings with Amazon Titan Text Embeddings V2")
        vehicles_with_embeddings = generate_embeddings(vehicles_with_context)

        # Step 4: Write LanceDB-formatted data to S3
        logger.info("Writing LanceDB data to S3")
        output_key = write_lancedb_to_s3(vehicles_with_embeddings)

        # Step 5: Write metadata
        metadata = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "environment": ENVIRONMENT,
            "vehicle_count": len(vehicles),
            "output_key": output_key,
            "source": "mocked_bigquery" if MOCK_BIGQUERY else "bigquery",
        }
        write_metadata_to_s3(metadata)

        logger.info("Data ingestion completed successfully: %s", output_key)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Data ingestion completed",
                    "vehicles_processed": len(vehicles),
                    "output_key": output_key,
                    "metadata": metadata,
                }
            ),
        }

    except (ClientError, json.JSONDecodeError, KeyError, ValueError, NotImplementedError) as e:
        logger.error("Error in data ingestion: %s", e, exc_info=True)
        raise


_DYNAMO_PARSERS: dict[str, Any] = {
    "S": lambda d: d["S"],
    "N": lambda d: float(d["N"]) if "." in d["N"] else int(d["N"]),
    "BOOL": lambda d: d["BOOL"],
    "NULL": lambda _: None,
    "SS": lambda d: d["SS"],
    "NS": lambda d: [float(n) if "." in n else int(n) for n in d["NS"]],
    "BS": lambda d: d["BS"],
    "M": lambda d: {k: parse_dynamodb_format(v) for k, v in d["M"].items()},
    "L": lambda d: [parse_dynamodb_format(item) for item in d["L"]],
}


def parse_dynamodb_format(data: Any) -> Any:
    """Parse DynamoDB format recursively to native Python types.

    Converts DynamoDB type markers:
    - {"N": "123"} -> 123 / {"S": "text"} -> "text"
    - {"BOOL": true} -> True / {"SS": [...]} -> [...]
    - {"M": {...}} -> {...} / {"L": [...]} -> [...]
    """
    if isinstance(data, dict):
        for key, parser in _DYNAMO_PARSERS.items():
            if key in data:
                return parser(data)
        return {k: parse_dynamodb_format(v) for k, v in data.items()}
    if isinstance(data, list):
        return [parse_dynamodb_format(item) for item in data]
    return data


def load_field_mapping_config() -> dict[str, Any]:
    """Load field mapping configuration from JSON file.

    Returns:
        Field mapping configuration dictionary
    """
    config_path = Path(__file__).parent / "config" / "field_mappings.json"

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info("Loaded field mapping config version %s", config.get("version", "unknown"))
        return config
    except FileNotFoundError:
        logger.warning("Field mapping config not found at %s, using defaults", config_path)
        # Return minimal default config
        return {
            "field_mappings": {
                "id": ["enquiry_id", "id", "vehicle_id"],
                "make": ["vehicle_make", "make", "manufacturer"],
                "model": ["vehicle_model", "model"],
                "year": ["manufacture_year", "year", "model_year"],
                "fuel_type": ["fuel", "fuel_type"],
                "body_type": ["body_category", "body_type", "vehicle_type"],
                "mileage": ["mileage", "odometer"],
                "price": ["reserve_price", "price"],
                "transmission": ["transmission", "gearbox"],
                "condition": ["condition_grade", "condition"],
            },
            "required_fields": ["id", "make", "model"],
            "fallback_values": {
                "year": "Unknown",
                "fuel_type": "unknown",
                "body_type": "vehicle",
                "condition": "good",
                "mileage": 0,
                "price": 0,
                "transmission": "unknown",
            },
        }


def resolve_field(vehicle: dict[str, Any], field_name: str, config: dict) -> Any:
    """Resolve a field value from vehicle data using config mappings.

    Tries each possible field name in order, returns first match.
    Falls back to default value if no match found.

    Args:
        vehicle: Vehicle data dictionary
        field_name: Standard field name to resolve
        config: Field mapping configuration

    Returns:
        Resolved field value or fallback
    """
    possible_names = config["field_mappings"].get(field_name, [field_name])

    # Try each possible field name
    for name in possible_names:
        if name in vehicle and vehicle[name] is not None:
            return vehicle[name]

    # Return fallback value if configured
    if field_name in config.get("fallback_values", {}):
        return config["fallback_values"][field_name]

    return None


def normalize_vehicle(vehicle: dict[str, Any], config: dict) -> dict[str, Any]:
    """Normalize a vehicle record to standard schema.

    Creates a new dict with standard field names, resolved from config.
    Preserves original fields in _original for debugging/audit trail.

    Args:
        vehicle: Raw vehicle data
        config: Field mapping configuration

    Returns:
        Normalized vehicle dictionary

    Raises:
        ValueError: If required fields are missing
    """
    normalized = {
        # Standard fields resolved from config
        "id": str(resolve_field(vehicle, "id", config)),
        "make": resolve_field(vehicle, "make", config),
        "model": resolve_field(vehicle, "model", config),
        "year": resolve_field(vehicle, "year", config),
        "fuel_type": resolve_field(vehicle, "fuel_type", config),
        "body_type": resolve_field(vehicle, "body_type", config),
        "mileage": resolve_field(vehicle, "mileage", config),
        "price": resolve_field(vehicle, "price", config),
        "transmission": resolve_field(vehicle, "transmission", config),
        "condition": resolve_field(vehicle, "condition", config),
        # Location fields (needed for filter_by_distance tool)
        "seller_latitude": resolve_field(vehicle, "seller_latitude", config),
        "seller_longitude": resolve_field(vehicle, "seller_longitude", config),
        "seller_city": resolve_field(vehicle, "seller_city", config),
        "seller_region": resolve_field(vehicle, "seller_region", config),
        # Preserve original fields for audit
        "_original": vehicle.copy(),
    }

    # Apply type conversions if configured
    type_conversions = config.get("type_conversions", {})
    for field, target_type in type_conversions.items():
        if field in normalized and normalized[field] is not None:
            try:
                if target_type == "int":
                    normalized[field] = int(normalized[field])
                elif target_type == "float":
                    normalized[field] = float(normalized[field])
                elif target_type == "string":
                    normalized[field] = str(normalized[field])
            except (ValueError, TypeError) as e:
                logger.warning("Could not convert %s to %s: %s", field, target_type, e)

    # Validate required fields
    required_fields = config.get("required_fields", ["id", "make", "model"])
    missing = [f for f in required_fields if not normalized.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return normalized


def fetch_sample_data_from_s3() -> list[dict[str, Any]]:
    """Fetch sample vehicle data from S3 (mocked BigQuery).

    Returns:
        List of vehicle dictionaries
    """
    try:
        response = s3_client.get_object(Bucket=DATA_BUCKET, Key=SAMPLE_DATA_KEY)
        data = json.loads(response["Body"].read().decode("utf-8"))
        vehicles = data.get("vehicles", [])

        # Parse DynamoDB format if present
        logger.info("Parsing %s vehicles from DynamoDB format", len(vehicles))
        parsed_vehicles = [parse_dynamodb_format(v) for v in vehicles]

        return parsed_vehicles
    except s3_client.exceptions.NoSuchKey:
        logger.warning("Sample data not found at %s, creating default dataset", SAMPLE_DATA_KEY)
        return create_default_vehicle_data()


def create_default_vehicle_data() -> list[dict[str, Any]]:
    """Create default sample vehicle data if none exists.

    Returns:
        List of sample vehicle dictionaries
    """
    vehicles = [
        {
            "id": "v001",
            "make": "BMW",
            "model": "3 Series",
            "year": 2019,
            "fuel_type": "diesel",
            "body_type": "sedan",
            "transmission": "automatic",
            "mileage": 25000,
            "price": 24000,
            "location": {"lat": 51.5074, "lon": -0.1278},  # London
            "condition": "excellent",
            "auction_id": f"auction_{datetime.now(tz=UTC).strftime('%Y_%m_%d')}",
        },
        {
            "id": "v002",
            "make": "Audi",
            "model": "Q5",
            "year": 2020,
            "fuel_type": "diesel",
            "body_type": "SUV",
            "transmission": "automatic",
            "mileage": 18000,
            "price": 32000,
            "location": {"lat": 51.4545, "lon": -2.5879},  # Bristol
            "condition": "excellent",
            "auction_id": f"auction_{datetime.now(tz=UTC).strftime('%Y_%m_%d')}",
        },
        # Add 3 more for demo
        {
            "id": "v003",
            "make": "Mercedes-Benz",
            "model": "C-Class",
            "year": 2018,
            "fuel_type": "petrol",
            "body_type": "sedan",
            "transmission": "automatic",
            "mileage": 35000,
            "price": 22000,
            "location": {"lat": 53.4808, "lon": -2.2426},  # Manchester
            "condition": "good",
            "auction_id": f"auction_{datetime.now(tz=UTC).strftime('%Y_%m_%d')}",
        },
    ]
    return vehicles


def contextualize_vehicles(vehicles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contextualize vehicle data with Amazon Nova Lite v1.

    Uses flexible field mapping configuration to normalize vehicle data
    from any source schema (auction marketplace, other providers, etc.).

    Args:
        vehicles: List of vehicle dictionaries (any format)

    Returns:
        Vehicles with contextualized descriptions and normalized fields
    """
    # Load field mapping configuration
    config = load_field_mapping_config()

    normalized_vehicles = []

    for vehicle in vehicles:
        try:
            # Normalize vehicle using flexible field resolution
            normalized = normalize_vehicle(vehicle, config)

            # Extract normalized fields for description generation
            vehicle_id = normalized["id"]
            year = normalized["year"]
            make = normalized["make"]
            model = normalized["model"]
            condition = normalized["condition"]
            mileage = normalized.get("mileage", 0)
            fuel_type = normalized["fuel_type"]
            body_type = normalized["body_type"]
            transmission = normalized["transmission"]

            # Generate contextualized description
            description = (
                f"{year} {make} {model} "
                f"in {condition} condition with {mileage:,} miles. "
                f"{fuel_type.capitalize()} {body_type} "
                f"with {transmission} transmission."
            )

            # For demo, we'll use the simple description
            # In production, you'd call Amazon Nova Lite v1 to enrich this
            normalized["contextualized_description"] = description

            normalized_vehicles.append(normalized)
            logger.info("Normalized vehicle %s: %s %s", vehicle_id, make, model)

        except ValueError as e:
            logger.warning("Skipping vehicle due to validation error: %s", e)
            # Log the problematic vehicle for debugging
            logger.warning("Problematic vehicle keys: %s", list(vehicle.keys()))
            continue

    logger.info("Successfully normalized %s/%s vehicles", len(normalized_vehicles), len(vehicles))

    if len(normalized_vehicles) == 0 and len(vehicles) > 0:
        raise ValueError("No vehicles passed validation. Check field mappings and required fields.")

    return normalized_vehicles


def generate_embeddings(vehicles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate embeddings for vehicle descriptions using Amazon Titan Text Embeddings V2.

    Args:
        vehicles: List of vehicles with contextualized descriptions

    Returns:
        Vehicles with embedding vectors
    """
    for vehicle in vehicles:
        description = vehicle["contextualized_description"]

        # Call Amazon Titan Text Embeddings V2
        try:
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps(
                    {
                        "inputText": description,
                    }
                ),
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]

            vehicle["embedding"] = embedding
            logger.info(
                "Generated embedding for vehicle %s: %s dimensions",
                vehicle["id"],
                len(embedding),
            )

        except (ClientError, json.JSONDecodeError, KeyError) as e:
            logger.error("Error generating embedding for vehicle %s: %s", vehicle["id"], e)
            # Skip vehicle — zero vectors poison semantic search results
            logger.warning("Skipping vehicle %s due to embedding failure", vehicle["id"])
            continue

    successful = [v for v in vehicles if "embedding" in v]
    failed_count = len(vehicles) - len(successful)
    if failed_count > 0:
        logger.warning("Embedding failures: %s/%s vehicles skipped", failed_count, len(vehicles))
    return successful


def write_lancedb_to_s3(vehicles: list[dict[str, Any]]) -> str:
    """Write LanceDB-formatted data to S3.

    Args:
        vehicles: List of vehicles with embeddings

    Returns:
        S3 key where data was written
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_key = f"lancedb/vehicles_{timestamp}.json"

    # Write to S3
    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key=output_key,
        Body=json.dumps({"vehicles": vehicles}, indent=2),
        ContentType="application/json",
    )

    # Also write to "latest" key for easy access
    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key="lancedb/latest.json",
        Body=json.dumps({"vehicles": vehicles}, indent=2),
        ContentType="application/json",
    )

    return output_key


def write_metadata_to_s3(metadata: dict[str, Any]) -> None:
    """Write pipeline metadata to S3.

    Args:
        metadata: Pipeline execution metadata
    """
    s3_client.put_object(
        Bucket=DATA_BUCKET,
        Key="metadata/last_refresh.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )
