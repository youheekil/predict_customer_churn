def go():
    model_type = ['Logistic Regression', 'Random Forest']
    for model in model_type:
            plt.rc('figure', figsize=(5, 5))
            if model == 'Logistic Regression':
                plt.text(0.01, 1.25, str('Logistic Regression Train'),
                        {'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.05, str(
                        classification_report(
                            y_train, y_train_preds_lr)), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.7, str(
                        classification_report(
                            y_test, y_test_preds_lr)), {
                        'fontsize': 10}, fontproperties='monospace')
            elif model == 'Random Forest':
                plt.text(0.01, 1.25, str('Random Forest Train'), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.05, str(
                        classification_report(
                            y_test, y_test_preds_rf)), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(0.01, 0.6, str('Random Forest Test'), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.7, str(
                        classification_report(
                            y_train, y_train_preds_rf)), {
                        'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(f"{constants.RESULTS_FILEPATH}{model}_report.png")
