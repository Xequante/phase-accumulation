from src.machine_learning.mplc_error import MPLCLoss


def mplc_compile(model, h, input_field, **kwargs):

    # Compile the model
    model.compile(optimizer='adam', 
                  loss=MPLCLoss(input_field=input_field, h=h),
                  metrics=['accuracy'])
