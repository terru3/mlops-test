###########
# BUILDER #
###########

# use base image for Lambda provided by AWS
# The image is built in two stages to keep the final image size minimal

FROM public.ecr.aws/lambda/python:3.10 as builder

RUN pip3 install --upgrade pip

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

#########
# FINAL #
#########
FROM public.ecr.aws/lambda/python:3.10
RUN pip3 install --upgrade pip

COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

# specify which function will be called when Lambda is invoked

CMD [ "logistic_model.infer_handle" ]

## previously: CMD [ "handler.handle" ]