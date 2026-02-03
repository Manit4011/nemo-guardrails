node("docker_aws_trivy_runner")
{
    def PROJECT_NAME                = "policy-bot-server"
    def SERVICE_PORT                = "5000"
    def GIT_REPO_URL                = "https://gitea.geminisolutions.com/Gemini-Solutions/PolicyBot.git"
    def DEPLOYMENT_BRANCH           = "flaskapp"

    def AWS_REGION                  = "ap-south-1"
    def AWS_ACCOUNT_ID              = "851725235990"
    def ECS_CLUSTER                 = "Cluster-dev"
    def ECS_SERVICE_NAME            = "Service-dev-${PROJECT_NAME}"
    def ECR_REPO_URL                = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
    def TASK_DEFINITION_FAMILY      = "task-definition-dev-${PROJECT_NAME}"
    def CONTAINER_NAME              = "Container-server"
    def ECS_ROLE                    = "ECS-task-excecution-Role"
    def ECS_TASK_ROLE               = "ECS-task-Role"
    def CODEDEPLOY_APP              = "Deploy-dev-${PROJECT_NAME}"
    def CODEDEPLOY_DEPLOYMENT_GROUP = "deployment-group-Deploy-dev-${PROJECT_NAME}"


    stage("Source code checkout")
    {
        cleanWs()
        dir(PROJECT_NAME)
        {
            checkout([$class: 'GitSCM' , branches: [[name: "*/${DEPLOYMENT_BRANCH}"]] ,  doGenerateSubmoduleConfigurations: false, extensions: [], submoduleCfg: [],  userRemoteConfigs: [[credentialsId: 'adminGitea', url: "${GIT_REPO_URL}"]]])
            sh "ls -lha"
        }
    }

    stage('ECR Login & Docker build') {
        dir(PROJECT_NAME)
        {
            container('docker-aws-trivy')
            {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws_cred_data_science']]) {
                echo "Logging in to AWS ECR..."
                sh """
                    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO_URL}
                """
                echo "Building Docker image"
                    sh """ docker build -t ${PROJECT_NAME}_$env.BUILD_NUMBER ."""
                    sh " docker tag ${PROJECT_NAME}_$env.BUILD_NUMBER ${ECR_REPO_URL}:$env.BUILD_NUMBER "
                echo "Pushing the Docker Image to ECR"
                    sh" docker push ${ECR_REPO_URL}:$env.BUILD_NUMBER "
                }
                
            }
        }
    }

    stage("Preparing Artifacts")
    {
        dir(PROJECT_NAME)
        {   
            sh "mkdir Artifacts"
            sh "mv appspec.yaml Artifacts/appspec.yaml && mv taskdef.json Artifacts/taskdef.json"
            echo "Changing directory to the Artifacts directory"
            sh "pwd && ls -lha"
            echo "Preparating artifacts"
            sh """
                        cd Artifacts
                        sed -i 's|<AWS_ACCOUNT_ID>|${AWS_ACCOUNT_ID}|g' taskdef.json
                        sed -i 's|<ECS_ROLE>|${ECS_ROLE}|g' taskdef.json
                        sed -i 's|<CONTAINER_NAME>|${CONTAINER_NAME}|g' taskdef.json
                        sed -i 's|<REPO_URL>|${ECR_REPO_URL}:$env.BUILD_NUMBER|g' taskdef.json
                        sed -i 's|<TASK_DEFINITION_FAMILY>|${TASK_DEFINITION_FAMILY}|g' taskdef.json
                        sed -i 's|<AWS_REGION>|${AWS_REGION}|g' taskdef.json
                        sed -i 's|<SERVICE_PORT>|${SERVICE_PORT}|g' taskdef.json
                        sed -i 's|<CONTAINER_NAME>|${CONTAINER_NAME}|g' appspec.yaml
                        sed -i 's|<SERVICE_PORT>|${SERVICE_PORT}|g' appspec.yaml
            """

        }
    }

    stage("ECS Deployment") {
        dir("${PROJECT_NAME}/Artifacts") {
            container('docker-aws-trivy')
            {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws_cred_data_science']]) {
                echo "Registering the ECS Task Definition..."
                sh "ls -lha"

                sh "cat taskdef.json"
                    
                script {
                    def registerTaskDefOutput = sh(
                        script: "aws ecs register-task-definition --cli-input-json file://taskdef.json --region ${AWS_REGION}",
                        returnStdout: true
                    ).trim()
                    
                    // Extract the Task Definition ARN from the output
                    def taskDefinitionArn = sh(
                        script: "echo '${registerTaskDefOutput}' | jq -r '.taskDefinition.taskDefinitionArn'",
                        returnStdout: true
                    ).trim()
                    echo "Task Definition ARN: ${taskDefinitionArn}"

                    // Save ARN to environment for later use
                    env.TASK_DEFINITION_ARN = taskDefinitionArn
                }

                sh '''
                    sed -i "s|<TASK_DEFINITION>|$TASK_DEFINITION_ARN|g" appspec.yaml
                '''

                sh "cat appspec.yaml"

                echo "REGISTERED TASK DEF"
                sh "ls -lha"

                echo 'Preparing revision.json with appspec.yaml content...'
                    sh '''
                        echo '{
                          "revisionType": "AppSpecContent",
                          "appSpecContent": {
                            "content": "'"$(cat appspec.yaml | sed ':a;N;$!ba;s/\\n/\\\\n/g')"'" 
                          }
                        }' > revision.json

                        echo 'Contents of revision.json:'
                        cat revision.json

                    '''
                        echo 'Creating deployment in CodeDeploy...'
                        sh """
                        aws deploy create-deployment \
                        --application-name ${CODEDEPLOY_APP} \
                        --deployment-group-name ${CODEDEPLOY_DEPLOYMENT_GROUP} \
                        --revision  file://revision.json\
                        --region ${AWS_REGION}
                        """
                
                }
            }
        
        }
    }

}