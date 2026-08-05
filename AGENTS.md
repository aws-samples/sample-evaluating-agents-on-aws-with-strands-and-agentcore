# Repository Agent Instructions

## Lambda Exposure Boundary

- MUST NOT create an AWS Lambda Function URL, including URLs configured with
  `AWS_IAM` authorization.
- MUST NOT grant public or account-principal access to invoke a Lambda
  function.
- Cross-service invocation is allowed only for a specific AWS service
  principal and a scoped source ARN. HTTP Lambda handlers MUST be invoked
  through the repository's authenticated Amazon API Gateway.
- Keep the app-wide CDK `LambdaInvokeBoundary` aspect enabled. Do not suppress
  or bypass its synthesis errors.
