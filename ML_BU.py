import sqlite3
from sqlite_ml.sqml import SQML
import pandas as pd
import pytermgui as ptg
import time
import sqlite_ml


def tests2(c1: ptg.Checkbox, c2: ptg.Checkbox):
    c1.checked = False
    c2.checked = False
    # exit(1)

def tests(l1: ptg.Label):
    l1.value = "asdasdasdsa"
    # exit(1)


def _select_feature(manager: ptg.WindowManager) -> None:
    import sqlite3
    from sqlite_ml.sqml import SQML
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    c_names = conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
    labels = [ptg.Label(name[0]) for name in c_names]

    container = ptg.Container(
        ptg.Label("[bold accent] Select feature/s"),
        ptg.Label(""),
        ptg.Label("[surface+1 dim italic] Possible features from the data: "),
        ptg.Label(""),
        ptg.Button("Submit button"),
    )
    for name in c_names:
        container.lazy_add(ptg.Label(name[0]))
    modal = ptg.Window("",
                       container
    ).center()
    modal.select(1)
    manager.add(modal)

    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)


if __name__ == '__main__':

    CONFIG = """
    config:
        InputField:
            styles:
                prompt: dim italic
                cursor: '@72'
        Label:
            styles:
                value: dim bold

        Window:
            styles:
                border: '60'
                corner: '60'

        Container:
            styles:
                border: '96'
                corner: '96'
    """

    with ptg.YamlLoader() as loader:
        loader.load(CONFIG)

    import sqlite3
    from sqlite_ml.sqml import SQML
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    c_names = conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
    with ptg.WindowManager() as manager:
        window = (
            ptg.Window(
                "",
                ptg.InputField("", prompt="Name of your ml model: "),
                "",
                ptg.Container(
                    ptg.Collapsible(
                        "Select your ml algorithmå",
                        "",
                        ptg.Splitter(
                            ptg.Container(
                                "[bold accent]Regressions",
                                {"linear_regression": [False]},
                                {"sgd": [False]},
                                {"ridge": [False]},
                                {"ridge_cv": [False]},
                                {"elastic_net": [False]},
                                {"elastic_net_cv": [False]},
                                {"lasso": [False]},
                                {"lasso_cv": [False]},
                                {"decision_tree": [False]},
                                {"ada_boost": [False]},
                                {"bagging": [False]},
                                {"gradient_boosting": [False]},
                                {"random_forest": [False]},
                                {"knn": [False]},
                                {"mlp": [False]},
                                {"svr": [False]},

                            ),
                            ptg.Container(
                                "[bold accent]Classifications",
                                {"logistic_regression": [False]},
                                {"sgd": [False]},
                                {"ridge": [False]},
                                {"ridge_cv": [False]},
                                {"decision_tree": [False]},
                                {"ada_boost": [False]},
                                {"bagging": [False]},
                                {"gradient_boosting": [False]},
                                {"random_forest": [False]},
                                {"knn": [False]},
                                {"mlp": [False]},
                                {"svc": [False]},

                            ),
                        ),

                    ),
                    box="EMPTY_VERTICAL",
                ),
                ptg.Container(
                    ptg.Collapsible(
                        "Select feature/target for your ml algorithm",
                        "",
                        ptg.Splitter(
                            ptg.Container(
                                "[bold accent]Features",
                                {"linear_regression": [False]},
                                {"sgd": [False]},
                                {"ridge": [False]},
                                {"ridge_cv": [False]},
                                {"elastic_net": [False]},
                                {"elastic_net_cv": [False]},
                                {"lasso": [False]},
                                {"lasso_cv": [False]},
                                {"decision_tree": [False]},
                                {"ada_boost": [False]},
                                {"bagging": [False]},
                                {"gradient_boosting": [False]},
                                {"random_forest": [False]},
                                {"knn": [False]},
                                {"mlp": [False]},
                                {"svr": [False]},

                            ),
                            ptg.Container(
                                "[bold accent]Target",
                                {"logistic_regression": [False]},
                                {"sgd": [False]},
                                {"ridge": [False]},
                                {"ridge_cv": [False]},
                                {"decision_tree": [False]},
                                {"ada_boost": [False]},
                                {"bagging": [False]},
                                {"gradient_boosting": [False]},
                                {"random_forest": [False]},
                                {"knn": [False]},
                                {"mlp": [False]},
                                {"svc": [False]},

                            ),
                        ),

                    ),
                    box="EMPTY_VERTICAL",
                ),

                "",
                ["Run Training", lambda *_: _select_feature(manager)],
                ["Run Prediction", lambda *_: _select_feature(manager)],
                width=100,
                overflow="auto",
                box="DOUBLE",
            )
            .set_title("[210 bold]ML workflow for Jube")
            .center()
        )

        manager.add(window)